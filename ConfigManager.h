// ConfigManager.h
#pragma once

#include <string>
#include <unordered_map>
#include <any>
#include <typeindex>
#include <filesystem>
#include <fstream>
#include <sstream>

class ConfigManager {
private:
    std::unordered_map<std::string, std::any> settings;
    std::string configFilePath;

public:
    ConfigManager(const std::string& configFile = "config.ini");

    template<typename T>
    T get(const std::string& key, const T& defaultValue) {
        if (settings.find(key) != settings.end()) {
            try {
                return std::any_cast<T>(settings[key]);
            } catch (const std::bad_any_cast&) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    template<typename T>
    void set(const std::string& key, const T& value) {
        settings[key] = value;
    }

    bool loadFromFile();
    bool saveToFile();
};