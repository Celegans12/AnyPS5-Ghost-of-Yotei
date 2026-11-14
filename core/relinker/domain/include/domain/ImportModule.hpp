#ifndef DOMAIN_IMPORTMODULE_HPP
#define DOMAIN_IMPORTMODULE_HPP

#include <domain/Types.hpp>
#include <algorithm>
#include <array>
#include <set>
#include <span>
#include <string_view>

namespace Domain {

inline constexpr std::array<std::string_view, 4> ModuleSuffixes = {".prx", ".sprx", ".suprx", ".debug_prx"};

inline std::string_view ModuleSuffix(const std::string_view name) {
    const auto suffix = std::find_if(ModuleSuffixes.begin(), ModuleSuffixes.end(), [&](const auto candidate) { return name.ends_with(candidate); });
    return suffix == ModuleSuffixes.end() ? std::string_view{} : *suffix;
}

inline std::string NeededLibraryName(std::string name) {
    const auto suffix = ModuleSuffix(name);
    if (!suffix.empty() && suffix != ".prx") name.replace(name.size() - suffix.size(), suffix.size(), ".prx");
    return name;
}

inline std::string ImportModuleName(std::string name) {
    name.resize(name.size() - ModuleSuffix(name).size());
    if (name.ends_with("-module")) name.resize(name.size() - 7);
    std::replace(name.begin(), name.end(), '.', '_');
    return name;
}

inline std::string DeclaredImportModule(const std::string& symbol, const std::map<std::uint64_t, std::string>& modules, const std::set<std::uint64_t>& ownModules = {}) {
    if (modules.empty()) return {};
    const auto first = symbol.find('#');
    if (first == std::string::npos) return {};
    const auto second = symbol.find('#', first + 1);
    if (second == std::string::npos || second + 1 == symbol.size())
        throw RelinkerException("Invalid qualified import: " + symbol);
    constexpr std::string_view alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    std::uint64_t id = 0;
    for (const char character : std::string_view(symbol).substr(second + 1)) {
        const auto digit = alphabet.find(character);
        if (digit == std::string_view::npos || id > (65535u - digit) / 64u)
            throw RelinkerException("Invalid import module ID: " + symbol);
        id = id * 64u + digit;
    }
    const auto module = modules.find(id);
    if (module == modules.end()) {
        if (ownModules.contains(id)) return {};
        throw RelinkerException("Unknown import module ID: " + symbol);
    }
    auto name = module->second;
    if (name.empty() || name.find_first_of("/\\:$\r\n") != std::string::npos)
        throw RelinkerException("Invalid import module name: " + name);
    return name;
}

inline std::string ImportLibrary(const std::string& symbol, const std::map<std::uint64_t, std::string>& libraries, std::span<const std::string> dependencies) {
    const auto first = symbol.find('#');
    const auto second = first == std::string::npos ? std::string::npos : symbol.find('#', first + 1);
    if (second == std::string::npos) return {};
    constexpr std::string_view alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    const auto encoded = std::string_view(symbol).substr(first + 1, second - first - 1);
    if (encoded.empty()) throw RelinkerException("Invalid import library ID: " + symbol);
    std::uint64_t id = 0;
    for (const char character : encoded) {
        const auto digit = alphabet.find(character);
        if (digit == std::string_view::npos || id > (65535u - digit) / 64u)
            throw RelinkerException("Invalid import library ID: " + symbol);
        id = id * 64u + digit;
    }
    const auto library = libraries.find(id);
    if (library == libraries.end()) throw RelinkerException("Unknown import library ID: " + symbol);
    auto name = library->second;
    if (name.empty() || name.find_first_of("/\\:$\r\n") != std::string::npos)
        throw RelinkerException("Invalid import library name: " + name);
    if (ModuleSuffix(name).empty()) name += ".prx";
    if (std::find(dependencies.begin(), dependencies.end(), name) != dependencies.end()) return name;
    std::string matched;
    for (const auto& dependency : dependencies) {
        if (ImportModuleName(dependency) != ImportModuleName(name)) continue;
        if (!matched.empty() && matched != dependency) throw RelinkerException("Ambiguous import library dependency: " + name);
        matched = dependency;
    }
    return matched;
}

inline std::string ImportModule(const std::string& symbol, const std::map<std::uint64_t, std::string>& modules,
    std::span<const std::string> dependencies = {}, const std::set<std::uint64_t>& ownModules = {},
    const std::map<std::uint64_t, std::string>& libraries = {}) {
    auto name = DeclaredImportModule(symbol, modules, ownModules);
    if (name.empty()) return {};
    if (ModuleSuffix(name).empty()) name += ".prx";
    if (std::find(dependencies.begin(), dependencies.end(), name) != dependencies.end()) return name;
    const auto normalized = ImportModuleName(name);
    const auto lookup = normalized == "libmonosgen" ? std::string("libmonosgen-2_0") : normalized;
    const auto findMatch = [&](const std::string& candidate) {
        std::string matched;
        for (const auto& dependency : dependencies) {
            if (ImportModuleName(dependency) != candidate) continue;
            if (!matched.empty() && matched != dependency) throw RelinkerException("Ambiguous import module dependency: " + name);
            matched = dependency;
        }
        return matched;
    };
    if (const auto matched = findMatch(normalized); !matched.empty()) return matched;
    if (lookup != normalized) {
        if (const auto matched = findMatch(lookup); !matched.empty()) return matched;
    }
    return libraries.empty() ? std::string{} : ImportLibrary(symbol, libraries, dependencies);
}

}

#endif
