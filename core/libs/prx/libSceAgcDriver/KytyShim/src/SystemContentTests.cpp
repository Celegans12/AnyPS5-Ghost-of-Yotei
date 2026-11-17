#include "prx/libSceAgcDriver/KytyShim/src/SystemContent.hpp"

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

using KytyShim::SystemContent::Param;
using KytyShim::SystemContent::ParseParamJson;

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAILED: %s\n", what);
    ++failures;
}

const Param* Find(const std::vector<Param>& params, std::string_view name) {
    for (const auto& param : params) {
        if (param.name == name) return &param;
    }
    return nullptr;
}

void ExpectText(const std::vector<Param>& params, std::string_view name, std::string_view value, const char* what) {
    const auto* param = Find(params, name);
    Expect(param != nullptr && param->isText && param->text == value, what);
}

void ExpectNumber(const std::vector<Param>& params, std::string_view name, std::int32_t value, const char* what) {
    const auto* param = Find(params, name);
    Expect(param != nullptr && !param->isText && param->number == value, what);
}

void FullTitle() {
    const auto parsed = ParseParamJson(R"({
        "applicationCategoryType": 0,
        "attribute": 4,
        "attribute3": 4194304,
        "contentId": "EP0000-TEST00453_00-0000000000000001",
        "contentVersion": "01.022.394",
        "titleId": "TEST00453",
        "userDefinedParam1": -1,
        "userDefinedParam2": 4294967295,
        "userDefinedParam3": 1.5,
        "localizedParameters": {
            "defaultLanguage": "fr-FR",
            "en-US": {"titleName": "Subnautica"},
            "fr-FR": {"titleName": "Subnautiça 😀"}
        },
        "kernel": {"flexibleMemorySize": 0}
    })");
    Expect(parsed.has_value(), "a PS5 param.json parses");
    if (!parsed.has_value()) return;
    ExpectText(*parsed, "TITLE_ID", "TEST00453", "titleId is TITLE_ID");
    ExpectText(*parsed, "CONTENT_ID", "EP0000-TEST00453_00-0000000000000001", "contentId is CONTENT_ID");
    ExpectText(*parsed, "APP_VER", "01.022.394", "contentVersion is APP_VER");
    ExpectNumber(*parsed, "ATTRIBUTE", 4, "attribute is ATTRIBUTE");
    ExpectNumber(*parsed, "ATTRIBUTE3", 4194304, "attribute3 is ATTRIBUTE3");
    ExpectNumber(*parsed, "USER_DEFINED_PARAM_1", -1, "a negative user parameter keeps its value");
    ExpectNumber(*parsed, "USER_DEFINED_PARAM_2", -1, "an unsigned user parameter keeps its low 32 bits, as Kyty's int32 cast");
    Expect(Find(*parsed, "USER_DEFINED_PARAM_3") == nullptr, "a real number is not an integer parameter");
    Expect(Find(*parsed, "USER_DEFINED_PARAM_4") == nullptr, "a missing user parameter is absent");
    ExpectText(*parsed, "TITLE", "Subnauti\xc3\xa7" "a \xf0\x9f\x98\x80", "the default language title, with escapes decoded to UTF-8");
    Expect(Find(*parsed, "TITLE_ID")->isText && Find(*parsed, "ATTRIBUTE") != nullptr && !Find(*parsed, "ATTRIBUTE")->isText, "strings and integers keep their kinds");
}

void Defaults() {
    const auto parsed = ParseParamJson("\xEF\xBB\xBF{\"titleId\": \"TEST00001\", \"contentVersion\": \"01.000\", \"appVersion\": \"02.000\", \"titleId\": \"TEST00002\","
                                       " \"localizedParameters\": {\"de-DE\": {\"titleName\": \"Erster\"}, \"en-US\": {\"titleName\": \"English\"}}}");
    Expect(parsed.has_value(), "a param.json with a byte order mark parses");
    if (!parsed.has_value()) return;
    ExpectText(*parsed, "TITLE_ID", "TEST00002", "a repeated key keeps its last value");
    ExpectText(*parsed, "APP_VER", "02.000", "appVersion overrides contentVersion");
    ExpectNumber(*parsed, "ATTRIBUTE", 0, "ATTRIBUTE defaults to 0");
    Expect(Find(*parsed, "ATTRIBUTE3") == nullptr, "a missing attribute3 is absent");
    Expect(Find(*parsed, "CONTENT_ID") == nullptr, "a missing contentId is absent");
    ExpectText(*parsed, "TITLE", "English", "without a default language, the en-US title");
    const auto first = ParseParamJson(R"({"localizedParameters": {"defaultLanguage": "it-IT", "ja-JP": {"titleName": "Ichi"}, "de-DE": {"titleName": "Zwei"}}})");
    Expect(first.has_value() && Find(*first, "TITLE") != nullptr && Find(*first, "TITLE")->text == "Ichi", "without the default language or en-US, the first title");
    const auto none = ParseParamJson(R"({"titleId": 5, "attribute": "4"})");
    Expect(none.has_value() && Find(*none, "TITLE_ID") == nullptr && Find(*none, "TITLE") == nullptr, "a titleId that is not a string is absent");
    Expect(none.has_value() && Find(*none, "ATTRIBUTE") != nullptr && Find(*none, "ATTRIBUTE")->number == 0, "an attribute that is not an integer leaves the default");
    const auto huge = ParseParamJson(R"({"attribute3": 18446744073709551616, "attribute": -9223372036854775808})");
    Expect(huge.has_value() && Find(*huge, "ATTRIBUTE3") == nullptr, "an integer beyond 64 bits is not an integer parameter");
    Expect(huge.has_value() && Find(*huge, "ATTRIBUTE") != nullptr && Find(*huge, "ATTRIBUTE")->number == 0, "the smallest int64 keeps its low 32 bits");
}

void Invalid() {
    Expect(!ParseParamJson("").has_value(), "an empty file is not a param.json");
    Expect(!ParseParamJson("[1, 2]").has_value(), "an array is not a param.json");
    Expect(!ParseParamJson(R"({"titleId": "TEST00001",})").has_value(), "a trailing comma is rejected");
    Expect(!ParseParamJson(R"({"titleId": "TEST00001"} x)").has_value(), "text after the document is rejected");
    Expect(!ParseParamJson(R"({"titleId": "\ud800"})").has_value(), "a lone surrogate is rejected");
    Expect(!ParseParamJson(R"({"attribute": 01})").has_value(), "a leading zero is rejected");
    Expect(!ParseParamJson("{\"titleId\": \"a\nb\"}").has_value(), "a raw control character in a string is rejected");
    std::string deep(100, '[');
    deep += std::string(100, ']');
    Expect(!ParseParamJson("{\"x\": " + deep + "}").has_value(), "nesting deeper than 64 levels is rejected");
}

}

int main() {
    FullTitle();
    Defaults();
    Invalid();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("kyty shim system content: all checks passed\n");
    return 0;
}
