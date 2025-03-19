#include "OutputFormatter.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "JointProcessor.h"

namespace fs = std::filesystem;

OutputFormatter::OutputFormatter() = default;
OutputFormatter::~OutputFormatter() = default;

bool OutputFormatter::saveToJson(const std::string& filename,
                              const std::vector<std::map<int, JointData>>& framesJointData,
                              const RecordingMetadata& metadata) {
    try {
        nlohmann::json outputJson;

        // Add metadata
        nlohmann::json metaJson;
        metaJson["recordingId"] = metadata.recordingId;
        metaJson["startDateTime"] = metadata.startDateTime;
        metaJson["totalFrames"] = metadata.totalFrames;
        metaJson["durationSeconds"] = metadata.durationSeconds;
        metaJson["averageFps"] = metadata.averageFps;
        metaJson["depthWidth"] = metadata.depthWidth;
        metaJson["depthHeight"] = metadata.depthHeight;
        metaJson["rgbWidth"] = metadata.rgbWidth;
        metaJson["rgbHeight"] = metadata.rgbHeight;

        // Add additional info
        for (const auto& [key, value] : metadata.additionalInfo) {
            metaJson[key] = value;
        }

        outputJson["metadata"] = metaJson;

        // Add frames data
        nlohmann::json framesJson = nlohmann::json::array();

        for (size_t frameIndex = 0; frameIndex < framesJointData.size(); frameIndex++) {
            nlohmann::json frameJson;
            frameJson["index"] = frameIndex;

            // Add timestamp if available
            if (frameIndex < metadata.frameTimestamps.size()) {
                frameJson["timestamp"] = metadata.frameTimestamps[frameIndex];

                // Calculate relative time if this is not the first frame
                if (frameIndex > 0 && !metadata.frameTimestamps.empty()) {
                    uint64_t startTs = metadata.frameTimestamps[0];
                    double relativeTimeMs = static_cast<double>(metadata.frameTimestamps[frameIndex] - startTs) / 1000000.0;
                    frameJson["relativeTimeMs"] = relativeTimeMs;
                }
            }

            // Add joints data
            nlohmann::json jointsJson = nlohmann::json::object();

            for (const auto& [jointId, jointData] : framesJointData[frameIndex]) {
                nlohmann::json jointJson;
                jointJson["position"] = {
                    {"x", jointData.x},
                    {"y", jointData.y},
                    {"z", jointData.z}
                };
                jointJson["confidence"] = jointData.confidence;

                // Use joint ID as the key
                jointsJson[std::to_string(jointId)] = jointJson;
            }

            frameJson["joints"] = jointsJson;
            framesJson.push_back(frameJson);
        }

        outputJson["frames"] = framesJson;

        // Create directories if needed
        fs::path filePath(filename);
        if (!filePath.parent_path().empty()) {
            fs::create_directories(filePath.parent_path());
        }

        // Write to file
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return false;
        }

        file << std::setw(2) << outputJson;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error saving JSON: " << e.what() << std::endl;
        return false;
    }
}

bool OutputFormatter::exportToCSV(const std::string& filename,
                               const std::vector<std::map<int, JointData>>& framesJointData,
                               const RecordingMetadata& metadata) {
    try {
        // Create directories if needed
        fs::path filePath(filename);
        if (!filePath.parent_path().empty()) {
            fs::create_directories(filePath.parent_path());
        }

        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return false;
        }

        // Create header
        file << "FrameIndex,Timestamp";

        // Find all joint IDs used across all frames
        std::set<int> allJointIds;
        for (const auto& frameJoints : framesJointData) {
            for (const auto& [jointId, _] : frameJoints) {
                allJointIds.insert(jointId);
            }
        }

        // Add columns for each joint
        for (int jointId : allJointIds) {
            file << ",Joint" << jointId << "_X,Joint" << jointId << "_Y,Joint" << jointId
                 << "_Z,Joint" << jointId << "_Confidence";
        }

        file << std::endl;

        // Write data for each frame
        for (size_t frameIndex = 0; frameIndex < framesJointData.size(); frameIndex++) {
            file << frameIndex;

            // Add timestamp if available
            if (frameIndex < metadata.frameTimestamps.size()) {
                file << "," << metadata.frameTimestamps[frameIndex];
            } else {
                file << ",0";
            }

            // Add data for each joint
            for (int jointId : allJointIds) {
                auto it = framesJointData[frameIndex].find(jointId);
                if (it != framesJointData[frameIndex].end()) {
                    const auto& jointData = it->second;
                    file << "," << jointData.x << "," << jointData.y << "," << jointData.z
                         << "," << jointData.confidence;
                } else {
                    file << ",0,0,0,0"; // Missing joint
                }
            }

            file << std::endl;
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error exporting to CSV: " << e.what() << std::endl;
        return false;
    }
}

bool OutputFormatter::exportToBinaryFormat(const std::string& filename,
                                        const std::vector<std::map<int, JointData>>& framesJointData,
                                        const RecordingMetadata& metadata) {
    try {
        // Create directories if needed
        fs::path filePath(filename);
        if (!filePath.parent_path().empty()) {
            fs::create_directories(filePath.parent_path());
        }

        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return false;
        }

        // Write metadata
        uint32_t recordingIdLen = static_cast<uint32_t>(metadata.recordingId.length());
        uint32_t startDateTimeLen = static_cast<uint32_t>(metadata.startDateTime.length());

        file.write(reinterpret_cast<const char*>(&recordingIdLen), sizeof(recordingIdLen));
        file.write(metadata.recordingId.c_str(), recordingIdLen);

        file.write(reinterpret_cast<const char*>(&startDateTimeLen), sizeof(startDateTimeLen));
        file.write(metadata.startDateTime.c_str(), startDateTimeLen);

        file.write(reinterpret_cast<const char*>(&metadata.totalFrames), sizeof(metadata.totalFrames));
        file.write(reinterpret_cast<const char*>(&metadata.durationSeconds), sizeof(metadata.durationSeconds));
        file.write(reinterpret_cast<const char*>(&metadata.averageFps), sizeof(metadata.averageFps));
        file.write(reinterpret_cast<const char*>(&metadata.depthWidth), sizeof(metadata.depthWidth));
        file.write(reinterpret_cast<const char*>(&metadata.depthHeight), sizeof(metadata.depthHeight));
        file.write(reinterpret_cast<const char*>(&metadata.rgbWidth), sizeof(metadata.rgbWidth));
        file.write(reinterpret_cast<const char*>(&metadata.rgbHeight), sizeof(metadata.rgbHeight));

        // Write number of additional info entries
        uint32_t additionalInfoCount = static_cast<uint32_t>(metadata.additionalInfo.size());
        file.write(reinterpret_cast<const char*>(&additionalInfoCount), sizeof(additionalInfoCount));

        // Write additional info
        for (const auto& [key, value] : metadata.additionalInfo) {
            uint32_t keyLen = static_cast<uint32_t>(key.length());
            uint32_t valueLen = static_cast<uint32_t>(value.length());

            file.write(reinterpret_cast<const char*>(&keyLen), sizeof(keyLen));
            file.write(key.c_str(), keyLen);

            file.write(reinterpret_cast<const char*>(&valueLen), sizeof(valueLen));
            file.write(value.c_str(), valueLen);
        }

        // Write frame count
        uint32_t frameCount = static_cast<uint32_t>(framesJointData.size());
        file.write(reinterpret_cast<const char*>(&frameCount), sizeof(frameCount));

        // Write frames data
        for (size_t frameIndex = 0; frameIndex < framesJointData.size(); frameIndex++) {
            // Write frame index
            uint32_t idx = static_cast<uint32_t>(frameIndex);
            file.write(reinterpret_cast<const char*>(&idx), sizeof(idx));

            // Write timestamp if available
            uint64_t timestamp = (frameIndex < metadata.frameTimestamps.size())
                ? metadata.frameTimestamps[frameIndex] : 0;
            file.write(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));

            // Write joint count
            uint32_t jointCount = static_cast<uint32_t>(framesJointData[frameIndex].size());
            file.write(reinterpret_cast<const char*>(&jointCount), sizeof(jointCount));

            // Write each joint
            for (const auto& [jointId, jointData] : framesJointData[frameIndex]) {
                file.write(reinterpret_cast<const char*>(&jointId), sizeof(jointId));
                file.write(reinterpret_cast<const char*>(&jointData.x), sizeof(jointData.x));
                file.write(reinterpret_cast<const char*>(&jointData.y), sizeof(jointData.y));
                file.write(reinterpret_cast<const char*>(&jointData.z), sizeof(jointData.z));
                file.write(reinterpret_cast<const char*>(&jointData.confidence), sizeof(jointData.confidence));
            }
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error exporting to binary format: " << e.what() << std::endl;
        return false;
    }
}

bool OutputFormatter::saveFrameJson(const std::string& directory,
                                 int frameIndex,
                                 const std::map<int, JointData>& joints,
                                 uint64_t timestamp) {
    try {
        nlohmann::json frameJson;
        frameJson["frame_index"] = frameIndex;
        frameJson["timestamp"] = timestamp;

        // Add joints data
        nlohmann::json jointsJson = nlohmann::json::object();

        for (const auto& [jointId, jointData] : joints) {
            nlohmann::json jointJson;
            jointJson["position"] = {
                {"x", jointData.x},
                {"y", jointData.y},
                {"z", jointData.z}
            };
            jointJson["confidence"] = jointData.confidence;

            jointsJson[std::to_string(jointId)] = jointJson;
        }

        frameJson["joints"] = jointsJson;

        // Create directories if needed
        fs::path dirPath(directory);
        fs::create_directories(dirPath);

        // Write to file
        fs::path filePath = dirPath / ("frame_" + std::to_string(frameIndex) + ".json");
        std::ofstream file(filePath.string());
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << filePath.string() << std::endl;
            return false;
        }

        file << std::setw(2) << frameJson;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error saving frame JSON: " << e.what() << std::endl;
        return false;
    }
}

// Helper function to create visualizations of joints data
cv::Mat OutputFormatter::createJointVisualization(const std::map<int, JointData>& joints,
                                              const cv::Mat& backgroundImage,
                                              const VisualizationOptions& options) {
    // Create a copy of the background image or a blank canvas
    cv::Mat result;
    if (!backgroundImage.empty()) {
        if (backgroundImage.type() == CV_8UC4) {
            cv::cvtColor(backgroundImage, result, cv::COLOR_BGRA2BGR);
        } else if (backgroundImage.type() == CV_8UC1) {
            cv::cvtColor(backgroundImage, result, cv::COLOR_GRAY2BGR);
        } else {
            result = backgroundImage.clone();
        }
    } else {
        // Create blank image with the specified size
        result = cv::Mat(options.height, options.width, CV_8UC3, cv::Scalar(40, 40, 40));
    }

    // Define joint connections for visualization
    static const std::vector<std::pair<int, int>> jointConnections = {
        {JointType_Head, JointType_Neck},
        {JointType_Neck, JointType_SpineShoulder},
        {JointType_SpineShoulder, JointType_SpineMid},
        {JointType_SpineMid, JointType_SpineBase},
        {JointType_SpineShoulder, JointType_ShoulderRight},
        {JointType_SpineShoulder, JointType_ShoulderLeft},
        {JointType_SpineBase, JointType_HipRight},
        {JointType_SpineBase, JointType_HipLeft},
        {JointType_ShoulderRight, JointType_ElbowRight},
        {JointType_ElbowRight, JointType_WristRight},
        {JointType_WristRight, JointType_HandRight},
        {JointType_HandRight, JointType_HandTipRight},
        {JointType_WristRight, JointType_ThumbRight},
        {JointType_ShoulderLeft, JointType_ElbowLeft},
        {JointType_ElbowLeft, JointType_WristLeft},
        {JointType_WristLeft, JointType_HandLeft},
        {JointType_HandLeft, JointType_HandTipLeft},
        {JointType_WristLeft, JointType_ThumbLeft},
        {JointType_HipRight, JointType_KneeRight},
        {JointType_KneeRight, JointType_AnkleRight},
        {JointType_AnkleRight, JointType_FootRight},
        {JointType_HipLeft, JointType_KneeLeft},
        {JointType_KneeLeft, JointType_AnkleLeft},
        {JointType_AnkleLeft, JointType_FootLeft}
    };

    // Draw connections
    for (const auto& [joint1, joint2] : jointConnections) {
        auto it1 = joints.find(joint1);
        auto it2 = joints.find(joint2);

        if (it1 != joints.end() && it2 != joints.end()) {
            const auto& j1 = it1->second;
            const auto& j2 = it2->second;

            // Skip if confidence is too low
            if (j1.confidence < options.minConfidence || j2.confidence < options.minConfidence) {
                continue;
            }

            // Convert 3D to 2D points
            cv::Point p1(static_cast<int>(j1.x), static_cast<int>(j1.y));
            cv::Point p2(static_cast<int>(j2.x), static_cast<int>(j2.y));

            // Check if points are within image bounds
            if (p1.x >= 0 && p1.x < result.cols && p1.y >= 0 && p1.y < result.rows &&
                p2.x >= 0 && p2.x < result.cols && p2.y >= 0 && p2.y < result.rows) {

                // Determine color based on confidence (green to red)
                float avgConfidence = (j1.confidence + j2.confidence) / 2.0f;
                cv::Scalar color;

                if (avgConfidence > 0.7f) {
                    color = options.highConfidenceColor;  // High confidence
                } else if (avgConfidence > 0.4f) {
                    color = options.mediumConfidenceColor;  // Medium confidence
                } else {
                    color = options.lowConfidenceColor;  // Low confidence
                }

                cv::line(result, p1, p2, color, options.lineThickness);
            }
        }
    }

    // Draw joints
    for (const auto& [jointId, joint] : joints) {
        // Skip if confidence is too low
        if (joint.confidence < options.minConfidence) {
            continue;
        }

        // Convert 3D to 2D point
        cv::Point p(static_cast<int>(joint.x), static_cast<int>(joint.y));

        // Check if point is within image bounds
        if (p.x >= 0 && p.x < result.cols && p.y >= 0 && p.y < result.rows) {
            // Determine color based on confidence
            cv::Scalar color;
            if (joint.confidence > 0.7f) {
                color = options.highConfidenceColor;
            } else if (joint.confidence > 0.4f) {
                color = options.mediumConfidenceColor;
            } else {
                color = options.lowConfidenceColor;
            }

            // Draw the joint as a circle
            cv::circle(result, p, options.jointRadius, color, -1);

            // Add joint ID if enabled
            if (options.showJointIds) {
                cv::putText(result, std::to_string(jointId),
                           cv::Point(p.x + 5, p.y - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }

            // Add depth info if enabled
            if (options.showDepthValues && joint.z > 0) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << joint.z << "m";
                cv::putText(result, ss.str(),
                           cv::Point(p.x + 5, p.y + 15),
                           cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
            }
        }
    }

    // Add frame information
    if (options.showFrameInfo && !options.frameInfo.empty()) {
        cv::putText(result, options.frameInfo,
                   cv::Point(20, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }

    return result;
}