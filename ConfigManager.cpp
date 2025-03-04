#include "ConfigManager.h"
#include <spdlog/spdlog.h>

ConfigManager::ConfigManager(const std::string& configFile)
    : configFilePath(configFile) {

    // Set default values
    settings["openpose_path"] = std::string("C:\\Users\\koqui\\OpenPose\\openpose\\bin\\OpenPoseDemo.exe");
    settings["net_resolution"] = 368;
    settings["use_maximum_accuracy"] = false;
    settings["keypoint_confidence_threshold"] = 40;
    settings["process_every_n_frames"] = 15;
    settings["recording_directory"] = std::string("recordings");
    settings["output_directory"] = std::string("processed");

    // Load config if file exists
    if (std::filesystem::exists(configFilePath)) {
        loadFromFile();
    } else {
        spdlog::info("Config file not found, using defaults");
        saveToFile(); // Create the file with defaults
    }
}

bool ConfigManager::loadFromFile() {
    try {
        std::ifstream file(configFilePath);
        if (!file.is_open()) {
            spdlog::error("Failed to open config file: {}", configFilePath);
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#' || line[0] == ';') {
                continue;
            }

            // Parse key=value pairs
            size_t delimPos = line.find('=');
            if (delimPos != std::string::npos) {
                std::string key = line.substr(0, delimPos);
                std::string value = line.substr(delimPos + 1);

                // Trim whitespace
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);

                // Store the value with the correct type
                if (key == "openpose_path" || key == "recording_directory" || key == "output_directory") {
                    settings[key] = value;
                } else if (key == "use_maximum_accuracy") {
                    settings[key] = (value == "true" || value == "1");
                } else {
                    // Try to parse as integer
                    try {
                        settings[key] = std::stoi(value);
                    } catch (...) {
                        // If not a valid integer, store as string
                        settings[key] = value;
                    }
                }
            }
        }

        spdlog::info("Loaded configuration from {}", configFilePath);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Error loading config file: {}", e.what());
        return false;
    }
}

bool ConfigManager::saveToFile() {
    try {
        std::ofstream file(configFilePath);
        if (!file.is_open()) {
            spdlog::error("Failed to open config file for writing: {}", configFilePath);
            return false;
        }

        file << "# KinectOpenPose Configuration File" << std::endl;
        file << "# Generated on " << __DATE__ << " " << __TIME__ << std::endl << std::endl;

        // Write all settings
        for (const auto& [key, value] : settings) {
            file << key << " = ";

            // Handle different types
            if (value.type() == typeid(std::string)) {
                file << std::any_cast<std::string>(value);
            } else if (value.type() == typeid(int)) {
                file << std::any_cast<int>(value);
            } else if (value.type() == typeid(bool)) {
                file << (std::any_cast<bool>(value) ? "true" : "false");
            } else if (value.type() == typeid(float) || value.type() == typeid(double)) {
                file << std::any_cast<double>(value);
            }

            file << std::endl;
        }

        spdlog::info("Saved configuration to {}", configFilePath);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Error saving config file: {}", e.what());
        return false;
    }
}