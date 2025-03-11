#include "Configuration.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <ranges>

Configuration::Configuration(std::filesystem::path configPath)
    : configFilePath(std::move(configPath)) {

    // Set default values for common settings
    settings["openpose_path"] = std::filesystem::path("bin/OpenPoseDemo.exe");
    settings["net_resolution"] = 368;
    settings["use_maximum_accuracy"] = false;
    settings["keypoint_confidence_threshold"] = 40;
    settings["process_every_n_frames"] = 15;
    settings["recording_directory"] = std::filesystem::path("recordings");
    settings["output_directory"] = std::filesystem::path("processed");
    settings["show_ui"] = true;
    settings["frame_rate_limit"] = 30;
    settings["enable_frame_limiting"] = true;
    settings["use_video_compression"] = true;

    // Try to load from file if it exists
    if (std::filesystem::exists(configFilePath)) {
        if (!loadFromFile(configFilePath)) {
            spdlog::warn("Failed to load configuration from {}, using defaults",
                        configFilePath.string());
        }
    } else {
        spdlog::info("Configuration file not found, creating with defaults");
    }
}

bool Configuration::loadFromFile(const std::filesystem::path& path) {
    try {
        std::ifstream file(path);
        if (!file.is_open()) {
            spdlog::error("Failed to open configuration file: {}", path.string());
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#' || line[0] == ';') {
                continue;
            }

            // Find the key-value separator
            size_t separatorPos = line.find('=');
            if (separatorPos == std::string::npos) {
                continue;  // Skip lines without key-value separator
            }

            // Extract key and value
            std::string key = trim(line.substr(0, separatorPos));
            std::string valueStr = trim(line.substr(separatorPos + 1));

            if (key.empty()) {
                continue;  // Skip if key is empty
            }

            // Parse and store the value
            settings[key] = parseValue(valueStr);
        }

        spdlog::info("Loaded configuration from {}", path.string());
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Error loading configuration file: {}", e.what());
        return false;
    }
}

bool Configuration::saveToFile(const std::filesystem::path& path) const {
    try {
        // Ensure the directory exists
        std::filesystem::path directory = path.parent_path();
        if (!directory.empty() && !std::filesystem::exists(directory)) {
            std::filesystem::create_directories(directory);
        }

        std::ofstream file(path);
        if (!file.is_open()) {
            spdlog::error("Failed to open configuration file for writing: {}", path.string());
            return false;
        }

        // Get current timestamp for header
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);

        // Write header
        file << "# KinectOpenPose Configuration File" << std::endl;
        file << "# Generated on " << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S")
             << std::endl << std::endl;

        // Write all settings in alphabetical order
        std::vector<std::string> keys = getKeys();
        std::ranges::sort(keys);

        for (const auto& key : keys) {
            auto it = settings.find(key);
            if (it != settings.end()) {
                file << key << " = " << valueToString(it->second) << std::endl;
            }
        }

        spdlog::info("Saved configuration to {}", path.string());
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Error saving configuration file: {}", e.what());
        return false;
    }
}

bool Configuration::reload() {
    if (!std::filesystem::exists(configFilePath)) {
        spdlog::error("Cannot reload: configuration file {} doesn't exist",
                      configFilePath.string());
        return false;
    }

    return loadFromFile(configFilePath);
}

void Configuration::setValue(const std::string& key, const ValueType& value) {
    settings[key] = value;
}

bool Configuration::hasValue(const std::string& key) const {
    return settings.contains(key);
}

bool Configuration::removeValue(const std::string& key) {
    return settings.erase(key) > 0;
}

void Configuration::clear() {
    settings.clear();
}

std::vector<std::string> Configuration::getKeys() const {
    std::vector<std::string> keys;
    keys.reserve(settings.size());

    for (const auto& key : settings | std::views::keys) {
        keys.push_back(key);
    }

    return keys;
}

std::string Configuration::toString() const {
    std::stringstream ss;
    ss << "Configuration (" << settings.size() << " items):" << std::endl;

    std::vector<std::string> keys = getKeys();
    std::ranges::sort(keys);

    for (const auto& key : keys) {
        auto it = settings.find(key);
        if (it != settings.end()) {
            ss << "  " << key << " = " << valueToString(it->second) << std::endl;
        }
    }

    return ss.str();
}

std::string Configuration::trim(const std::string& str) {
    const auto begin = std::ranges::find_if_not(str,
                                                [](unsigned char c) { return std::isspace(c); });

    const auto end = std::find_if_not(str.rbegin(), str.rend(),
        [](unsigned char c) { return std::isspace(c); }).base();

    return (begin < end) ? std::string(begin, end) : std::string();
}

Configuration::ValueType Configuration::parseValue(const std::string& value) {
    // Handle boolean values
    if (value == "true" || value == "yes" || value == "1") {
        return true;
    }
    if (value == "false" || value == "no" || value == "0") {
        return false;
    }

    // Try to parse as integer
    try {
        size_t pos;
        int intValue = std::stoi(value, &pos);

        // If the entire string was parsed, return an integer
        if (pos == value.length()) {
            return intValue;
        }
    } catch (...) {
        // Not an integer, continue to other types
    }

    // Try to parse as double
    try {
        size_t pos;
        double doubleValue = std::stod(value, &pos);

        // If the entire string was parsed, return a double
        if (pos == value.length()) {
            return doubleValue;
        }
    } catch (...) {
        // Not a double, continue to other types
    }

    // Check if it might be a path (contains path separators or extensions)
    if (value.find('/') != std::string::npos ||
        value.find('\\') != std::string::npos ||
        value.find('.') != std::string::npos) {
        return std::filesystem::path(value);
    }

    // Default to string
    return value;
}

std::string Configuration::valueToString(const ValueType& value) {
    return std::visit([](auto&& arg) -> std::string {
        using T = std::decay_t<decltype(arg)>;

        if constexpr (std::is_same_v<T, std::string>) {
            return arg;
        } else if constexpr (std::is_same_v<T, int>) {
            return std::to_string(arg);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::to_string(arg);
        } else if constexpr (std::is_same_v<T, bool>) {
            return arg ? "true" : "false";
        } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
            return arg.string();
        } else {
            return "[unknown type]";
        }
    }, value);
}