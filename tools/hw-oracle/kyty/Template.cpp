#include "Template.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace KytyOracle {
namespace {

struct Replacement {
    std::uint32_t index;
    std::array<std::uint32_t, 2> words;
};

constexpr std::uint32_t ScalarNop = 0xbf800000u;

constexpr std::array<Replacement, 2> PrologueReplacements {{
    {0, {ScalarNop, ScalarNop}},
    {5, {0xe0381000u, 0x80020401u}},
}};

constexpr std::array<Replacement, 4> EpilogueReplacements {{
    {0, {0xe0781000u, 0x80000a02u}},
    {2, {0xe0781010u, 0x80000e02u}},
    {4, {0xe0781020u, 0x80001202u}},
    {6, {0xe0781030u, 0x80001602u}},
}};

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

void RequireWords(std::span<const std::uint32_t> actual, std::span<const std::uint32_t> expected, std::uint32_t first,
                  const char* part) {
    for (std::uint32_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error(std::string("the kernel's ") + part + " is not tools/hw-oracle/template.s: word " +
                                     std::to_string(first + i) + " is " + Hex(actual[i]) + ", expected " + Hex(expected[i]));
        }
    }
}

template <std::size_t N>
void Replace(std::span<std::uint32_t> words, const std::array<Replacement, N>& replacements) {
    for (const auto& replacement : replacements) {
        std::ranges::copy(replacement.words, words.begin() + replacement.index);
    }
}

}

Program RewriteTemplate(std::span<const std::uint32_t> text) {
    const auto fixed = static_cast<std::uint32_t>(TemplatePrologue.size() + TemplateEpilogue.size());
    if (text.size() < fixed) {
        throw std::runtime_error("the kernel has " + std::to_string(text.size()) + " words, fewer than the template's " +
                                 std::to_string(fixed));
    }
    Program program;
    program.code.assign(text.begin(), text.end());
    program.body_begin = static_cast<std::uint32_t>(TemplatePrologue.size());
    program.body_end = static_cast<std::uint32_t>(text.size() - TemplateEpilogue.size());
    RequireWords(text.first(TemplatePrologue.size()), TemplatePrologue, 0, "prologue");
    RequireWords(text.last(TemplateEpilogue.size()), TemplateEpilogue, program.body_end, "epilogue");
    Replace(std::span(program.code).first(TemplatePrologue.size()), PrologueReplacements);
    Replace(std::span(program.code).last(TemplateEpilogue.size()), EpilogueReplacements);
    return program;
}

}
