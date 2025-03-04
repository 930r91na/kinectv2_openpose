// SkeletonExporter.cpp
#include "SkeletonExporter.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <unordered_map>
#include <spdlog/spdlog.h>
#include <filesystem>

// Use a mutex for file access to prevent multithreading issues
static std::mutex fileMutex;

// Map to keep track of frame counts for each file
static std::unordered_map<std::string, int> fileFrameCounts;

bool SkeletonExporter::saveTo3DSkeletonFile(const std::vector<Person3D>& people, const std::string& outputPath) {
    std::lock_guard<std::mutex> lock(fileMutex);

    try {
        std::ofstream file(outputPath);
        if (!file.is_open()) {
            spdlog::error("Failed to open file for writing: {}", outputPath);
            return false;
        }

        file << "SKELETON_FILE_V1.0" << std::endl;
        file << "NumFrames: 1" << std::endl;
        file << "Frame: 0" << std::endl;
        file << "NumPersons: " << people.size() << std::endl;

        for (size_t personIdx = 0; personIdx < people.size(); personIdx++) {
            const auto& person = people[personIdx];
            file << "Person: " << personIdx << std::endl;
            file << "NumKeypoints: " << person.keypoints.size() << std::endl;

            for (size_t i = 0; i < person.keypoints.size(); i++) {
                const auto& kp = person.keypoints[i];
                file << i << " "
                     << std::fixed << std::setprecision(6)
                     << kp.x << " " << kp.y << " " << kp.z << " " << kp.confidence
                     << std::endl;
            }
        }

        file.close();
        spdlog::info("Successfully saved skeleton data to {}", outputPath);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Error saving skeleton file: {}", e.what());
        return false;
    }
}

bool SkeletonExporter::createSkeletonFile(const std::string& outputPath) {
    std::lock_guard<std::mutex> lock(fileMutex);

    try {
        // Make sure directory exists
        std::filesystem::path path(outputPath);
        std::filesystem::create_directories(path.parent_path());

        std::ofstream file(outputPath);
        if (!file.is_open()) {
            spdlog::error("Failed to create skeleton file: {}", outputPath);
            return false;
        }

        // Write header
        file << "SKELETON_FILE_V1.0" << std::endl;
        file << "NumFrames: 0" << std::endl;  // Will be updated later

        file.close();

        // Initialize frame count
        fileFrameCounts[outputPath] = 0;

        spdlog::debug("Created new skeleton file: {}", outputPath);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Error creating skeleton file: {}", e.what());
        return false;
    }
}

bool SkeletonExporter::appendFrameToSkeletonFile(const std::vector<Person3D>& people,
                                                const std::string& outputPath,
                                                uint64_t timestamp) {
    std::lock_guard<std::mutex> lock(fileMutex);

    try {
        // Check if file exists, create if not
        if (!std::filesystem::exists(outputPath)) {
            if (!createSkeletonFile(outputPath)) {
                return false;
            }
        }

        // Open file in append mode
        std::ofstream file(outputPath, std::ios::app);
        if (!file.is_open()) {
            spdlog::error("Failed to open skeleton file for appending: {}", outputPath);
            return false;
        }

        // Get current frame index
        int frameIdx = fileFrameCounts[outputPath];

        // Write frame header
        file << "Frame: " << frameIdx << std::endl;
        file << "Timestamp: " << timestamp << std::endl;
        file << "NumPersons: " << people.size() << std::endl;

        // Write person data
        for (size_t personIdx = 0; personIdx < people.size(); personIdx++) {
            const auto& person = people[personIdx];
            file << "Person: " << personIdx << std::endl;
            file << "NumKeypoints: " << person.keypoints.size() << std::endl;

            for (size_t i = 0; i < person.keypoints.size(); i++) {
                const auto& kp = person.keypoints[i];
                file << i << " "
                     << std::fixed << std::setprecision(6)
                     << kp.x << " " << kp.y << " " << kp.z << " " << kp.confidence
                     << std::endl;
            }
        }

        file.close();

        // Increment frame count
        fileFrameCounts[outputPath]++;

        return true;
    } catch (const std::exception& e) {
        spdlog::error("Error appending to skeleton file: {}", e.what());
        return false;
    }
}

bool SkeletonExporter::finalizeSkeletonFile(const std::string& outputPath) {
    std::lock_guard<std::mutex> lock(fileMutex);

    try {
        // Get frame count
        int frameCount = fileFrameCounts[outputPath];

        // Read entire file
        std::ifstream inFile(outputPath);
        if (!inFile.is_open()) {
            spdlog::error("Failed to open skeleton file for finalizing: {}", outputPath);
            return false;
        }

        std::string content;
        std::string line;
        bool headerUpdated = false;

        while (std::getline(inFile, line)) {
            if (!headerUpdated && line.find("NumFrames:") != std::string::npos) {
                // Update frame count
                content += "NumFrames: " + std::to_string(frameCount) + "\n";
                headerUpdated = true;
            } else {
                content += line + "\n";
            }
        }

        inFile.close();

        // Write updated content
        std::ofstream outFile(outputPath);
        if (!outFile.is_open()) {
            spdlog::error("Failed to open skeleton file for writing: {}", outputPath);
            return false;
        }

        outFile << content;
        outFile.close();

        spdlog::info("Finalized skeleton file with {} frames: {}", frameCount, outputPath);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Error finalizing skeleton file: {}", e.what());
        return false;
    }
}

bool SkeletonExporter::saveToCSV(const std::vector<Person3D>& people, const std::string& outputPath) {
    std::lock_guard<std::mutex> lock(fileMutex);

    try {
        std::ofstream file(outputPath);
        if (!file.is_open()) {
            spdlog::error("Failed to open CSV file for writing: {}", outputPath);
            return false;
        }

        // Write header
        file << "person_id,keypoint_id,x,y,z,confidence" << std::endl;

        // Write data
        for (size_t personIdx = 0; personIdx < people.size(); personIdx++) {
            const auto& person = people[personIdx];

            for (size_t i = 0; i < person.keypoints.size(); i++) {
                const auto& kp = person.keypoints[i];
                file << personIdx << ","
                     << i << ","
                     << std::fixed << std::setprecision(6)
                     << kp.x << ","
                     << kp.y << ","
                     << kp.z << ","
                     << kp.confidence
                     << std::endl;
            }
        }

        file.close();
        spdlog::info("Successfully saved CSV data to {}", outputPath);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Error saving CSV file: {}", e.what());
        return false;
    }
}