#include "loader/systemContent.h"

#include "prx/libSceAgcDriver/KytyShim/src/SystemContent.hpp"
#include "prx/libc/include/General.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace KytyShim::SystemContent {

namespace {

constexpr const char* ParamJsonGuestPath = "/app0/sce_sys/param.json";

struct Json {
    enum class Kind { Null, Boolean, Integer, Real, String, Array, Object };
    Kind kind = Kind::Null;
    bool negative = false;
    std::uint64_t magnitude = 0;
    std::string text;
    std::vector<Json> items;
    std::vector<std::pair<std::string, Json>> members;

    const Json* Find(std::string_view key) const {
        if (kind != Kind::Object) return nullptr;
        for (auto member = members.rbegin(); member != members.rend(); ++member) {
            if (member->first == key) return &member->second;
        }
        return nullptr;
    }
};

class Parser {
public:
    explicit Parser(std::string_view source) : _source(source) {}

    std::optional<Json> Document() {
        if (_source.substr(0, 3) == "\xEF\xBB\xBF") _at = 3;
        Json value;
        if (!Value(value, 0)) return std::nullopt;
        Space();
        if (_at != _source.size()) return std::nullopt;
        return value;
    }

private:
    void Space() {
        while (_at < _source.size() && (_source[_at] == ' ' || _source[_at] == '\t' || _source[_at] == '\n' || _source[_at] == '\r')) ++_at;
    }

    bool Literal(std::string_view word) {
        if (_source.substr(_at, word.size()) != word) return false;
        _at += word.size();
        return true;
    }

    static void AppendUtf8(std::string& out, std::uint32_t code) {
        if (code < 0x80) {
            out.push_back(static_cast<char>(code));
        } else if (code < 0x800) {
            out.push_back(static_cast<char>(0xc0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
        } else if (code < 0x10000) {
            out.push_back(static_cast<char>(0xe0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
        } else {
            out.push_back(static_cast<char>(0xf0 | (code >> 18)));
            out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
        }
    }

    bool Hex4(std::uint32_t& code) {
        if (_source.size() - _at < 4) return false;
        code = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = _source[_at++];
            code <<= 4;
            if (c >= '0' && c <= '9') code |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') code |= static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') code |= static_cast<std::uint32_t>(c - 'A' + 10);
            else return false;
        }
        return true;
    }

    bool String(std::string& out) {
        if (_at >= _source.size() || _source[_at] != '"') return false;
        ++_at;
        while (_at < _source.size()) {
            const char c = _source[_at++];
            if (c == '"') return true;
            if (static_cast<unsigned char>(c) < 0x20) return false;
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (_at >= _source.size()) return false;
            const char escape = _source[_at++];
            switch (escape) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                std::uint32_t code = 0;
                if (!Hex4(code)) return false;
                if (code >= 0xd800 && code <= 0xdbff) {
                    std::uint32_t low = 0;
                    if (!Literal("\\u") || !Hex4(low) || low < 0xdc00 || low > 0xdfff) return false;
                    code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
                } else if (code >= 0xdc00 && code <= 0xdfff) {
                    return false;
                }
                AppendUtf8(out, code);
                break;
            }
            default: return false;
            }
        }
        return false;
    }

    bool Number(Json& out) {
        const auto start = _at;
        if (_at < _source.size() && _source[_at] == '-') ++_at;
        const auto digits = _at;
        if (_at >= _source.size() || _source[_at] < '0' || _source[_at] > '9') return false;
        if (_source[_at] == '0') {
            ++_at;
        } else {
            while (_at < _source.size() && _source[_at] >= '0' && _source[_at] <= '9') ++_at;
        }
        bool real = false;
        if (_at < _source.size() && _source[_at] == '.') {
            real = true;
            ++_at;
            if (_at >= _source.size() || _source[_at] < '0' || _source[_at] > '9') return false;
            while (_at < _source.size() && _source[_at] >= '0' && _source[_at] <= '9') ++_at;
        }
        if (_at < _source.size() && (_source[_at] == 'e' || _source[_at] == 'E')) {
            real = true;
            ++_at;
            if (_at < _source.size() && (_source[_at] == '+' || _source[_at] == '-')) ++_at;
            if (_at >= _source.size() || _source[_at] < '0' || _source[_at] > '9') return false;
            while (_at < _source.size() && _source[_at] >= '0' && _source[_at] <= '9') ++_at;
        }
        out.text.assign(_source.substr(start, _at - start));
        if (real) {
            out.kind = Json::Kind::Real;
            return true;
        }
        std::uint64_t magnitude = 0;
        for (auto i = digits; i < _at; ++i) {
            const auto digit = static_cast<std::uint64_t>(_source[i] - '0');
            if (magnitude > (UINT64_MAX - digit) / 10) {
                out.kind = Json::Kind::Real;
                return true;
            }
            magnitude = magnitude * 10 + digit;
        }
        out.negative = digits != start;
        if (out.negative && magnitude > static_cast<std::uint64_t>(INT64_MAX) + 1) {
            out.kind = Json::Kind::Real;
            return true;
        }
        out.kind = Json::Kind::Integer;
        out.magnitude = magnitude;
        return true;
    }

    bool Value(Json& out, int depth) {
        if (depth > 64) return false;
        Space();
        if (_at >= _source.size()) return false;
        switch (_source[_at]) {
        case '{': {
            ++_at;
            out.kind = Json::Kind::Object;
            Space();
            if (_at < _source.size() && _source[_at] == '}') {
                ++_at;
                return true;
            }
            for (;;) {
                Space();
                std::string key;
                if (!String(key)) return false;
                Space();
                if (_at >= _source.size() || _source[_at] != ':') return false;
                ++_at;
                Json member;
                if (!Value(member, depth + 1)) return false;
                out.members.emplace_back(std::move(key), std::move(member));
                Space();
                if (_at >= _source.size()) return false;
                if (_source[_at] == ',') {
                    ++_at;
                    continue;
                }
                if (_source[_at] != '}') return false;
                ++_at;
                return true;
            }
        }
        case '[': {
            ++_at;
            out.kind = Json::Kind::Array;
            Space();
            if (_at < _source.size() && _source[_at] == ']') {
                ++_at;
                return true;
            }
            for (;;) {
                Json item;
                if (!Value(item, depth + 1)) return false;
                out.items.push_back(std::move(item));
                Space();
                if (_at >= _source.size()) return false;
                if (_source[_at] == ',') {
                    ++_at;
                    continue;
                }
                if (_source[_at] != ']') return false;
                ++_at;
                return true;
            }
        }
        case '"':
            out.kind = Json::Kind::String;
            return String(out.text);
        case 't':
            out.kind = Json::Kind::Boolean;
            out.magnitude = 1;
            return Literal("true");
        case 'f':
            out.kind = Json::Kind::Boolean;
            return Literal("false");
        case 'n':
            out.kind = Json::Kind::Null;
            return Literal("null");
        default:
            return Number(out);
        }
    }

    std::string_view _source;
    std::size_t _at = 0;
};

class Table {
public:
    explicit Table(std::vector<Param>& params) : _params(params) {}

    void Text(const char* name, const std::string& value) {
        auto& param = Slot(name);
        param.isText = true;
        param.text = value;
    }

    void Number(const char* name, std::int32_t value) {
        auto& param = Slot(name);
        param.isText = false;
        param.text.clear();
        param.number = value;
    }

private:
    Param& Slot(const char* name) {
        for (auto& param : _params) {
            if (param.name == name) return param;
        }
        auto& added = _params.emplace_back();
        added.name = name;
        return added;
    }

    std::vector<Param>& _params;
};

class ParamSfo {
public:
    ParamSfo() {
        const auto path = ResolvePath_nid_no_patch(ParamJsonGuestPath);
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            Report(path, "cannot be opened");
            return;
        }
        const std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        auto parsed = ParseParamJson(content);
        if (!parsed.has_value()) {
            Report(path, content.empty() ? "is empty" : "is not a JSON object");
            return;
        }
        _params = std::move(*parsed);
        _opened = true;
    }

    const Param* Find(const char* name) const {
        if (!_opened) return nullptr;
        for (const auto& param : _params) {
            if (param.name == name) return &param;
        }
        return nullptr;
    }

private:
    static void Report(const std::filesystem::path& path, const char* problem) {
        std::fprintf(stderr, "[kyty] %s (%s) %s: the Kyty core sees no param.sfo values (no TITLE_ID: no Vulkan pipeline cache, no per-title shortcuts)\n", ParamJsonGuestPath, path.c_str(), problem);
        std::fflush(stderr);
    }

    std::vector<Param> _params;
    bool _opened = false;
};

const ParamSfo& Params() {
    static const ParamSfo params;
    return params;
}

}

std::optional<std::vector<Param>> ParseParamJson(std::string_view content) {
    const auto document = Parser(content).Document();
    if (!document.has_value() || document->kind != Json::Kind::Object) return std::nullopt;
    const Json& root = *document;
    std::vector<Param> params;
    Table table(params);
    const auto addString = [&](const char* key, const char* name) {
        if (const Json* value = root.Find(key); value != nullptr && value->kind == Json::Kind::String) table.Text(name, value->text);
    };
    const auto addInteger = [&](const char* key, const char* name) {
        const Json* value = root.Find(key);
        if (value == nullptr || value->kind != Json::Kind::Integer) return;
        const auto bits = value->negative ? ~value->magnitude + 1 : value->magnitude;
        table.Number(name, static_cast<std::int32_t>(static_cast<std::uint32_t>(bits)));
    };
    addString("titleId", "TITLE_ID");
    addString("contentId", "CONTENT_ID");
    addString("contentVersion", "APP_VER");
    addString("appVersion", "APP_VER");
    table.Number("ATTRIBUTE", 0);
    addInteger("attribute", "ATTRIBUTE");
    addInteger("attribute3", "ATTRIBUTE3");
    addInteger("userDefinedParam1", "USER_DEFINED_PARAM_1");
    addInteger("userDefinedParam2", "USER_DEFINED_PARAM_2");
    addInteger("userDefinedParam3", "USER_DEFINED_PARAM_3");
    addInteger("userDefinedParam4", "USER_DEFINED_PARAM_4");
    std::string defaultLanguage;
    std::vector<std::pair<std::string, std::string>> titles;
    if (const Json* localized = root.Find("localizedParameters"); localized != nullptr && localized->kind == Json::Kind::Object) {
        if (const Json* language = localized->Find("defaultLanguage"); language != nullptr && language->kind == Json::Kind::String) defaultLanguage = language->text;
        for (const auto& [language, entry] : localized->members) {
            if (entry.kind != Json::Kind::Object) continue;
            if (const Json* title = entry.Find("titleName"); title != nullptr && title->kind == Json::Kind::String) titles.emplace_back(language, title->text);
        }
    }
    std::string title;
    for (const auto& [language, name] : titles) {
        if (!defaultLanguage.empty() && language == defaultLanguage) {
            title = name;
            break;
        }
    }
    if (title.empty()) {
        for (const auto& [language, name] : titles) {
            if (language == "en-US") {
                title = name;
                break;
            }
        }
    }
    if (title.empty() && !titles.empty()) title = titles.front().second;
    if (!title.empty()) table.Text("TITLE", title);
    return params;
}

}

namespace Loader {

bool SystemContentParamSfoGetInt(const char* name, int32_t* value) {
    if (name == nullptr || value == nullptr) return false;
    const auto* param = KytyShim::SystemContent::Params().Find(name);
    if (param == nullptr || param->isText) return false;
    *value = param->number;
    return true;
}

bool SystemContentParamSfoGetString(const char* name, std::string* value) {
    if (name == nullptr || value == nullptr) return false;
    const auto* param = KytyShim::SystemContent::Params().Find(name);
    if (param == nullptr || !param->isText) return false;
    *value = param->text;
    return true;
}

bool SystemContentParamSfoGetString(const char* name, char* value, size_t value_size) {
    if (name == nullptr || value == nullptr) return false;
    const auto* param = KytyShim::SystemContent::Params().Find(name);
    if (param == nullptr || !param->isText || param->text.size() + 1 > value_size) return false;
    std::memcpy(value, param->text.c_str(), param->text.size() + 1);
    return true;
}

}
