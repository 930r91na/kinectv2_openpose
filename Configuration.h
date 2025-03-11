#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <optional>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <memory>
#include <vector>
#include <iostream>
#include <type_traits>

/**
 * @brief Modern configuration system for the application
 * 
 * This class provides a type-safe, easy to use configuration system
 * that supports various data types, default values, and loading/saving
 * from/to files.
 */
class Configuration {
public:
    // Configuration value types supported
    using ValueType = std::variant<
        std::string,
        int,
        double,
        bool,
        std::filesystem::path
    >;
    
    // Constructor with optional config file
    explicit Configuration(std::filesystem::path configPath = "config.ini");
    
    // Load/save methods
    bool loadFromFile(const std::filesystem::path& path);
    bool saveToFile(const std::filesystem::path& path) const;
    
    // Reload from the original file
    bool reload();
    
    // Value access with type safety
    template<typename T>
    std::optional<T> getValue(const std::string& key) const {
        auto it = settings.find(key);
        if (it == settings.end()) {
            return std::nullopt;
        }
        
        try {
            return std::get<T>(it->second);
        } catch (const std::bad_variant_access&) {
            return std::nullopt;
        }
    }
    
    // Value retrieval with default fallback
    template<typename T>
    T getValueOr(const std::string& key, const T& defaultValue) const {
        auto value = getValue<T>(key);
        return value.value_or(defaultValue);
    }
    
    // Path retrieval with default fallback and optional checking
    std::filesystem::path getPathOr(const std::string& key, 
                                   const std::filesystem::path& defaultValue,
                                   bool checkExists = false) const {
        auto path = getValueOr<std::filesystem::path>(key, defaultValue);
        
        if (checkExists && !std::filesystem::exists(path)) {
            return defaultValue;
        }
        
        return path;
    }
    
    // Settings access
    void setValue(const std::string& key, const ValueType& value);
    bool hasValue(const std::string& key) const;
    bool removeValue(const std::string& key);
    
    // Clear all settings
    void clear();
    
    // Get all keys
    std::vector<std::string> getKeys() const;
    
    // String representation for debugging
    std::string toString() const;

    const std::filesystem::path& getConfigFilePath() const noexcept {
        return configFilePath;
    }

    
private:
    std::unordered_map<std::string, ValueType> settings;
    std::filesystem::path configFilePath;
    
    // Helper methods for parsing
    static std::string trim(const std::string& str);
    static ValueType parseValue(const std::string& value);
    static std::string valueToString(const ValueType& value);
};

// Template specialization for handling paths
template<>
inline std::optional<std::filesystem::path> Configuration::getValue<std::filesystem::path>(
    const std::string& key) const {
    
    auto it = settings.find(key);
    if (it == settings.end()) {
        return std::nullopt;
    }
    
    try {
        // Handle string to path conversion
        if (std::holds_alternative<std::string>(it->second)) {
            return std::filesystem::path(std::get<std::string>(it->second));
        }
        // Direct path access
        return std::get<std::filesystem::path>(it->second);
    } catch (const std::bad_variant_access&) {
        return std::nullopt;
    }
}