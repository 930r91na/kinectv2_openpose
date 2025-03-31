#include "JointProcessorAnalyzer.h"

#include <filesystem>
#include <fstream>
#include <vector>
#include <kinect.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <chrono>
#include <regex>
#include <iomanip>

#include "spdlog/spdlog.h"

namespace fs = std::filesystem;

JointProcessorAnalyzer::JointProcessorAnalyzer() :
    dataLoaded(false),
    dataProcessed(false),
    metadataLoaded(false)
{
    recordingPath = "";
    depthRawPath = "";
    jsonPath = "";
    processingTempPath = "";

    spdlog::info("JointProcessorAnalyzer initialized");
}

JointProcessorAnalyzer::~JointProcessorAnalyzer()
{
    frames.clear();
    frameTimestamps.clear();

    spdlog::info("JointProcessorAnalyzer destroyed");
}

bool JointProcessorAnalyzer::loadRecording(const std::filesystem::path& recordingPath, const std::filesystem::path& processedPath) {
    this->recordingPath = recordingPath;
    this->processedPath = processedPath;
    dataLoaded = false;
    dataProcessed = false;
    metadataLoaded = false;
    frames.clear();
    frameTimestamps.clear();

    // Verify directory exists
    if (!std::filesystem::exists(recordingPath)) {
        spdlog::error("Recording directory not found: {}", recordingPath.string());
        return false;
    }

    // Set up paths to important subdirectories
    depthRawPath = recordingPath / "depth_raw";
    processingTempPath = recordingPath / "processing_temp";

    // Find the JSON directory - check multiple possible locations
    if (std::filesystem::exists(processedPath / "json")) {
        jsonPath = processedPath / "json";
        spdlog::info("Using JSON data from processed path: {}", jsonPath.string());
    } else if (std::filesystem::exists(recordingPath / "json")) {
        jsonPath = recordingPath / "json";
        spdlog::info("Using JSON data from recording path: {}", jsonPath.string());
    } else {
        spdlog::warn("No json directory found in recording or processed paths");
        // We'll still try to load JSON files from the main directory
    }

    // Load metadata
    if (!loadMetadataFromRecording()) {
        spdlog::warn("Could not load complete metadata, using defaults");
    }

    // Load timestamps from CSV
    std::filesystem::path timestampPath = recordingPath / "frame_timestamps.csv";
    if (!std::filesystem::exists(timestampPath)) {
        spdlog::error("Timestamp file not found: {}", timestampPath.string());
        return false;
    }

    try {
        // Read timestamps
        std::ifstream timestampFile(timestampPath);
        std::string line;

        // Skip header
        std::getline(timestampFile, line);

        // Keep track of processed frames
        int totalFrames = 0;
        int framesWithJointData = 0;

        // Initialize previous timestamp for delta calculation
        uint64_t prevTimestamp = 0;

        // Process each line in the timestamps file
        while (std::getline(timestampFile, line)) {
            std::stringstream ss(line);
            std::string token;

            // Parse frame index
            std::getline(ss, token, ',');
            int frameIndex = std::stoi(token);

            // Parse timestamp
            std::getline(ss, token, ',');
            uint64_t timestamp = std::stoull(token);

            // Store timestamp
            frameTimestamps.push_back(timestamp);

            // Calculate delta from previous frame if not the first frame
            uint64_t deltaTime = 0;
            if (prevTimestamp > 0) {
                deltaTime = timestamp - prevTimestamp;
            }
            prevTimestamp = timestamp;

            // Create frame data entry
            FrameData frame;
            frame.frameIndex = frameIndex;
            frame.timestamp = timestamp;

            // Try to load joint data for this frame from JSON
            nlohmann::json frameJson;
            bool hasJointData = false;

            if (loadJsonData(frameIndex, frameJson)) {
                // Process joint data if available
                if (frameJson.contains("people") && frameJson["people"].is_array() && !frameJson["people"].empty()) {
                    const auto& person = frameJson["people"][0];
                    if (person.contains("pose_keypoints_3d") && person["pose_keypoints_3d"].is_array()) {
                        const auto& keypoints = person["pose_keypoints_3d"];

                        // Make sure we have enough keypoints (x, y, z, confidence groupings)
                        if (keypoints.size() >= 4 && keypoints.size() % 4 == 0) {
                            hasJointData = true;
                            framesWithJointData++;

                            // Process each keypoint (x, y, z, confidence)
                            for (size_t i = 0; i < keypoints.size(); i += 4) {
                                // Extract the values, ensuring they're converted to float
                                float x = keypoints[i].get<double>();
                                float y = keypoints[i+1].get<double>();
                                float z = keypoints[i+2].get<double>();
                                float confidence = keypoints[i+3].get<double>();

                                // Map OpenPose index to Kinect joint ID
                                int jointId = mapOpenPoseToKinectJoint(i / 4);
                                if (jointId >= 0) {
                                    // Store raw joint data
                                    frame.rawJoints[jointId] = cv::Point3f(x, y, z);
                                    frame.confidences[jointId] = confidence;
                                }
                            }
                        }
                    }
                }
            }

            if (!hasJointData) {
                spdlog::debug("No joint data found for frame {}", frameIndex);
            }

            frames.push_back(frame);
            totalFrames++;

            // Log progress occasionally
            if (totalFrames % 1000 == 0) {
                spdlog::info("Loaded {} frames from timestamps", totalFrames);
            }
        }

        metadata.totalFrames = frames.size();
        if (frames.size() >= 2) {
            // Calculate duration and FPS
            uint64_t duration_ns = frames.back().timestamp - frames.front().timestamp;
            metadata.durationSeconds = duration_ns / 1e9;
            metadata.averageFps = frames.size() / metadata.durationSeconds;
        }

        spdlog::info("Loaded {} frames from recording ({} with joint data)", frames.size(), framesWithJointData);

        if (framesWithJointData == 0) {
            spdlog::warn("No frames with joint data were found! Check JSON paths and formats");
        } else if (framesWithJointData < frames.size() / 2) {
            spdlog::warn("Less than half of the frames have joint data ({}%). Check JSON paths and formats",
                         (framesWithJointData * 100.0f) / frames.size());
        }

        dataLoaded = true;
        return true;
    }
    catch (const std::exception& e) {
        spdlog::error("Error loading recording data: {}", e.what());
        return false;
    }
}
bool JointProcessorAnalyzer::loadMetadataFromRecording() {
    // Try to extract recording ID from path
    std::string recordingId = recordingPath.filename().string();
    metadata.recordingId = recordingId;

    // Load metadata from metadata.txt
    fs::path metadataPath = recordingPath / "metadata.txt";
    if (fs::exists(metadataPath)) {
        try {
            std::ifstream metadataFile(metadataPath);
            std::string line;

            std::regex startTimeRegex("Recording started at: (.+)");
            std::regex sessionIdRegex("Session ID: (.+)");

            while (std::getline(metadataFile, line)) {
                std::smatch match;
                if (std::regex_search(line, match, startTimeRegex) && match.size() > 1) {
                    metadata.startDateTime = match[1];
                } else if (std::regex_search(line, match, sessionIdRegex) && match.size() > 1) {
                    metadata.recordingId = match[1];
                } else {
                    // Parse key-value pairs for additional info
                    size_t separatorPos = line.find(": ");
                    if (separatorPos != std::string::npos) {
                        std::string key = line.substr(0, separatorPos);
                        std::string value = line.substr(separatorPos + 2);
                        metadata.additionalInfo[key] = value;
                    }
                }
            }

            // Try to parse video_timing.txt for more metadata
            fs::path timingPath = recordingPath / "video_timing.txt";
            if (fs::exists(timingPath)) {
                std::ifstream timingFile(timingPath);
                while (std::getline(timingFile, line)) {
                    // Look for FPS and duration information
                    if (line.find("Actual FPS:") != std::string::npos) {
                        size_t pos = line.find("Actual FPS: ");
                        if (pos != std::string::npos) {
                            try {
                                metadata.averageFps = std::stof(line.substr(pos + 12));
                            } catch(...) {}
                        }
                    } else if (line.find("Duration:") != std::string::npos) {
                        size_t pos = line.find("Duration: ");
                        if (pos != std::string::npos) {
                            size_t endPos = line.find(" seconds");
                            if (endPos != std::string::npos) {
                                try {
                                    metadata.durationSeconds = std::stod(line.substr(pos + 10, endPos - (pos + 10)));
                                } catch(...) {}
                            }
                        }
                    }
                }
            }

            metadataLoaded = true;
            return true;
        } catch (const std::exception& e) {
            spdlog::error("Error loading metadata: {}", e.what());
            return false;
        }
    }

    return false;
}

// to handle confidence scores and proper filtering
bool JointProcessorAnalyzer::processData() {
    if (frames.empty()) {
        spdlog::error("No frames to process");
        return false;
    }

    spdlog::info("Processing {} frames with enhanced joint filter", frames.size());

    // Reset the joint processor
    jointProcessor.reset();

    // First pass - process each frame
    for (size_t i = 0; i < frames.size(); i++) {
        FrameData& frame = frames[i];

        // Process each joint in the frame
        for (const auto& joint : frame.rawJoints) {
            int jointId = joint.first;
            const cv::Point3f& rawPos = joint.second;

            // Get confidence if available
            float confidence = 0.5f;  // Default mid confidence
            auto confIt = frame.confidences.find(jointId);
            if (confIt != frame.confidences.end()) {
                confidence = confIt->second;
            }

            // Store confidence for processing
            jointProcessor.storeJointConfidence(jointId, confidence);

            // Look ahead for future positions if available
            std::vector<cv::Point3f> futurePositions;
            for (size_t j = i + 1; j < std::min(frames.size(), i + 4); j++) {
                auto futureIt = frames[j].rawJoints.find(jointId);
                if (futureIt != frames[j].rawJoints.end()) {
                    futurePositions.push_back(futureIt->second);
                }
            }

            // Calculate time interval to next frame
            float timeInterval = 1.0f/30.0f;  // Default 30 fps
            if (i + 1 < frames.size()) {
                uint64_t currentTs = frame.timestamp;
                uint64_t nextTs = frames[i + 1].timestamp;
                if (nextTs > currentTs) {
                    // Convert from nanoseconds to seconds
                    timeInterval = static_cast<float>((nextTs - currentTs) / 1000000000.0);
                }
            }

            // Filter the joint position
            cv::Point3f filteredPos = jointProcessor.getFilteredJointPosition(
                jointId, rawPos,
                futurePositions.empty() ? nullptr : &futurePositions,
                &frame.rawJoints, timeInterval);

            // Store the filtered position
            frame.filteredJoints[jointId] = filteredPos;
        }
    }

    spdlog::info("Joint filtering complete");
    dataProcessed = true;
    return true;
}

cv::Mat JointProcessorAnalyzer::visualizeSideBySide(int frameIndex, const cv::Mat& background) {
    // Validate frame index
    if (frameIndex < 0 || frameIndex >= static_cast<int>(frames.size())) {
        return cv::Mat();
    }

    // Create background or use provided one
    cv::Mat canvas;
    if (!background.empty()) {
        canvas = background.clone();
        // Ensure we have a 3-channel BGR image
        if (canvas.channels() == 1) {
            cv::cvtColor(canvas, canvas, cv::COLOR_GRAY2BGR);
        } else if (canvas.channels() == 4) {
            cv::cvtColor(canvas, canvas, cv::COLOR_BGRA2BGR);
        }
    } else {
        canvas = cv::Mat(720, 1280, CV_8UC3, cv::Scalar(40, 40, 40));
    }

    // Split canvas into left and right halves
    cv::Mat leftHalf = canvas(cv::Rect(0, 0, canvas.cols/2, canvas.rows));
    cv::Mat rightHalf = canvas(cv::Rect(canvas.cols/2, 0, canvas.cols/2, canvas.rows));

    // Add titles
    cv::putText(leftHalf, "Raw Joint Data", cv::Point(20, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2);

    cv::putText(rightHalf, "Filtered Joint Data", cv::Point(20, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2);

    // Add frame index
    std::string frameText = "Frame: " + std::to_string(frameIndex);
    cv::putText(leftHalf, frameText, cv::Point(20, 60),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 1);
    cv::putText(rightHalf, frameText, cv::Point(20, 60),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 1);

    // Try to load depth image if available
    cv::Mat depthImage;
    if (loadDepthData(frameIndex, depthImage)) {
        // Convert depth to visible image
        cv::Mat depthVis;
        cv::normalize(depthImage, depthVis, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(depthVis, depthVis, cv::COLORMAP_JET);

        // Resize and make semi-transparent
        cv::resize(depthVis, depthVis, cv::Size(leftHalf.cols, leftHalf.rows));
        cv::addWeighted(leftHalf, 0.7, depthVis, 0.3, 0, leftHalf);
        cv::addWeighted(rightHalf, 0.7, depthVis, 0.3, 0, rightHalf);
    }

    // Draw raw skeleton on left half
    drawSkeleton(leftHalf, frames[frameIndex].rawJoints, cv::Scalar(255, 0, 0), false);

    // Draw filtered skeleton on right half
    drawSkeleton(rightHalf, frames[frameIndex].filteredJoints, cv::Scalar(0, 255, 0), true);

    // Add confidence information
    int yPos = 90;
    for (const auto& joint : frames[frameIndex].confidences) {
        if (joint.second > 0.1f) {  // Only show joints with some confidence
            std::string jointName = getJointName(joint.first);
            std::string confText = jointName + ": " + std::to_string(static_cast<int>(joint.second * 100)) + "%";

            // Show confidences on both sides
            cv::putText(leftHalf, confText, cv::Point(20, yPos),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

            yPos += 20;
            if (yPos > leftHalf.rows - 30) break;  // Prevent overflow
        }
    }

    return canvas;
}

// Calculate velocity between two joint positions
cv::Point3f JointProcessorAnalyzer::calculateJointVelocity(const cv::Point3f& pos1, const cv::Point3f& pos2, float timeInterval) {
    // Prevent division by zero
    if (timeInterval <= 0.0001f) {
        timeInterval = 0.0001f;
    }

    // Calculate velocity vector
    return (pos2 - pos1) / timeInterval;
}

// Calculate angle between three joints (in degrees)
float JointProcessorAnalyzer::calculateJointAngle(int jointId1, int jointId2, int jointId3, const std::map<int, cv::Point3f>& joints) {
    auto it1 = joints.find(jointId1);
    auto it2 = joints.find(jointId2);
    auto it3 = joints.find(jointId3);

    if (it1 == joints.end() || it2 == joints.end() || it3 == joints.end()) {
        return 0.0f; // Can't calculate angle
    }

    const cv::Point3f& p1 = it1->second;
    const cv::Point3f& p2 = it2->second;
    const cv::Point3f& p3 = it3->second;

    // Create vectors
    cv::Point3f v1 = p1 - p2;
    cv::Point3f v2 = p3 - p2;

    // Calculate angle
    float dotProduct = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    float v1Mag = std::sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
    float v2Mag = std::sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);

    // Prevent division by zero
    if (v1Mag < 0.0001f || v2Mag < 0.0001f) {
        return 0.0f;
    }

    float cosAngle = dotProduct / (v1Mag * v2Mag);

    // Clamp to valid range for acos
    cosAngle = std::max(-1.0f, std::min(1.0f, cosAngle));

    // Convert to degrees
    return std::acos(cosAngle) * (180.0f / 3.14159265358979323846f);
}

// Draw skeleton on image
void JointProcessorAnalyzer::drawSkeleton(cv::Mat& image, const std::map<int, cv::Point3f>& joints, cv::Scalar color, bool isFiltered) {
    // Define bone connections for visualization
    static const std::vector<std::pair<int, int>> bones = {
        // Torso
        {JointType_Head, JointType_Neck},
        {JointType_Neck, JointType_SpineShoulder},
        {JointType_SpineShoulder, JointType_SpineMid},
        {JointType_SpineMid, JointType_SpineBase},
        {JointType_SpineShoulder, JointType_ShoulderRight},
        {JointType_SpineShoulder, JointType_ShoulderLeft},
        {JointType_SpineBase, JointType_HipRight},
        {JointType_SpineBase, JointType_HipLeft},

        // Right Arm
        {JointType_ShoulderRight, JointType_ElbowRight},
        {JointType_ElbowRight, JointType_WristRight},
        {JointType_WristRight, JointType_HandRight},
        {JointType_HandRight, JointType_HandTipRight},
        {JointType_WristRight, JointType_ThumbRight},

        // Left Arm
        {JointType_ShoulderLeft, JointType_ElbowLeft},
        {JointType_ElbowLeft, JointType_WristLeft},
        {JointType_WristLeft, JointType_HandLeft},
        {JointType_HandLeft, JointType_HandTipLeft},
        {JointType_WristLeft, JointType_ThumbLeft},

        // Right Leg
        {JointType_HipRight, JointType_KneeRight},
        {JointType_KneeRight, JointType_AnkleRight},
        {JointType_AnkleRight, JointType_FootRight},

        // Left Leg
        {JointType_HipLeft, JointType_KneeLeft},
        {JointType_KneeLeft, JointType_AnkleLeft},
        {JointType_AnkleLeft, JointType_FootLeft}
    };

    // Project 3D points to 2D space
    std::map<int, cv::Point> points2D;
    for (const auto& joint : joints) {
        if (joint.second.x == 0 && joint.second.y == 0 && joint.second.z == 0) {
            continue;  // Skip invalid joints
        }

        // Simple projection (assuming z is depth)
        // You might need to adjust this based on your coordinate system
        float scale = 1.0f;
        if (joint.second.z > 0) {
            scale = 2.0f / joint.second.z;  // Scale based on depth
        }

        int x = static_cast<int>(joint.second.x * scale + image.cols/2);
        int y = static_cast<int>(joint.second.y * scale + image.rows/2);

        // Ensure point is within image bounds
        if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
            points2D[joint.first] = cv::Point(x, y);
        }
    }

    // Draw bones
    for (const auto& bone : bones) {
        auto it1 = points2D.find(bone.first);
        auto it2 = points2D.find(bone.second);

        if (it1 != points2D.end() && it2 != points2D.end()) {
            cv::line(image, it1->second, it2->second, color, isFiltered ? 2 : 1);
        }
    }

    // Draw joints
    for (const auto& point : points2D) {
        cv::circle(image, point.second, isFiltered ? 4 : 3, color, -1);
    }
}

// Get joint name from joint ID
std::string JointProcessorAnalyzer::getJointName(int jointId) {
    switch (jointId) {
        case JointType_SpineBase: return "Spine Base";
        case JointType_SpineMid: return "Spine Mid";
        case JointType_Neck: return "Neck";
        case JointType_Head: return "Head";
        case JointType_ShoulderLeft: return "Left Shoulder";
        case JointType_ElbowLeft: return "Left Elbow";
        case JointType_WristLeft: return "Left Wrist";
        case JointType_HandLeft: return "Left Hand";
        case JointType_ShoulderRight: return "Right Shoulder";
        case JointType_ElbowRight: return "Right Elbow";
        case JointType_WristRight: return "Right Wrist";
        case JointType_HandRight: return "Right Hand";
        case JointType_HipLeft: return "Left Hip";
        case JointType_KneeLeft: return "Left Knee";
        case JointType_AnkleLeft: return "Left Ankle";
        case JointType_FootLeft: return "Left Foot";
        case JointType_HipRight: return "Right Hip";
        case JointType_KneeRight: return "Right Knee";
        case JointType_AnkleRight: return "Right Ankle";
        case JointType_FootRight: return "Right Foot";
        case JointType_SpineShoulder: return "Spine Shoulder";
        case JointType_HandTipLeft: return "Left Hand Tip";
        case JointType_ThumbLeft: return "Left Thumb";
        case JointType_HandTipRight: return "Right Hand Tip";
        case JointType_ThumbRight: return "Right Thumb";
        default: return "Joint " + std::to_string(jointId);
    }
}

bool JointProcessorAnalyzer::loadDepthData(int frameIndex, cv::Mat& depthMat) {
    if (frameIndex < 0 || frameIndex >= static_cast<int>(frames.size())) {
        return false;
    }

    // Construct path to depth file
    std::filesystem::path depthPath = depthRawPath / ("frame_" + std::to_string(frameIndex) + ".bin");

    if (!std::filesystem::exists(depthPath)) {
        return false;
    }

    try {
        std::ifstream depthFile(depthPath, std::ios::binary);
        if (!depthFile.is_open()) {
            return false;
        }

        // Read dimensions
        int rows = 0, cols = 0;
        depthFile.read(reinterpret_cast<char*>(&rows), sizeof(int));
        depthFile.read(reinterpret_cast<char*>(&cols), sizeof(int));

        // Validate dimensions
        if (rows <= 0 || cols <= 0 || rows > 10000 || cols > 10000) {
            return false;
        }

        // Create the depth matrix
        depthMat = cv::Mat(rows, cols, CV_16UC1);

        // Read the data
        depthFile.read(reinterpret_cast<char*>(depthMat.data),
                      depthMat.elemSize() * depthMat.total());

        return true;
    }
    catch (const std::exception& e) {
        spdlog::error("Error loading depth data: {}", e.what());
        return false;
    }
}

bool JointProcessorAnalyzer::loadJsonData(int frameIndex, nlohmann::json& jsonData) {

    // Construct the path to the JSON file based on frame index
    std::string filename = "frame_" + std::to_string(frameIndex) + ".json";
    std::filesystem::path jsonFilePath = jsonPath / filename;

    if (!exists(jsonFilePath)) {
        // Try alternative path formats if the primary one doesn't exist
        jsonFilePath = processedPath / "json" / ("frame_" + std::to_string(frameIndex) + ".json");

        if (!std::filesystem::exists(jsonFilePath)) {
            jsonFilePath = recordingPath / "json" / ("frame_" + std::to_string(frameIndex) + ".json");

            if (!std::filesystem::exists(jsonFilePath)) {
                return false;
            }
        }
    }

    try {
        // Read the JSON file
        std::ifstream file(jsonFilePath);
        if (!file.is_open()) {
            return false;
        }

        file >> jsonData;
        return true;
    } catch (const std::exception& e) {
        // Log error but don't throw it further
        spdlog::error("Error reading JSON file {}: {}", jsonFilePath.string(), e.what());
        return false;
    }
}

bool JointProcessorAnalyzer::exportToJson(const std::filesystem::path& outputPath) {
    if (!dataLoaded) {
        spdlog::error("Cannot export to JSON: No data loaded");
        return false;
    }

    try {
        nlohmann::json outputJson;

        // Add metadata
        nlohmann::json metadataJson;
        metadataJson["recordingId"] = metadata.recordingId;
        metadataJson["startDateTime"] = metadata.startDateTime;
        metadataJson["totalFrames"] = metadata.totalFrames;
        metadataJson["durationSeconds"] = metadata.durationSeconds;
        metadataJson["averageFps"] = metadata.averageFps;

        // Add additional metadata
        for (const auto& [key, value] : metadata.additionalInfo) {
            metadataJson[key] = value;
        }

        outputJson["metadata"] = metadataJson;

        // Add frames array
        outputJson["frames"] = nlohmann::json::array();

        // Loop through all frames
        for (const auto& frame : frames) {
            nlohmann::json frameJson;
            frameJson["frameIndex"] = frame.frameIndex;
            frameJson["timestamp"] = frame.timestamp;

            // Add raw joints
            nlohmann::json rawJointsJson = nlohmann::json::object();
            for (const auto& [jointId, jointPos] : frame.rawJoints) {
                nlohmann::json jointJson;
                jointJson["x"] = jointPos.x;
                jointJson["y"] = jointPos.y;
                jointJson["z"] = jointPos.z;

                // Add confidence if available
                auto confIt = frame.confidences.find(jointId);
                if (confIt != frame.confidences.end()) {
                    jointJson["confidence"] = confIt->second;
                } else {
                    jointJson["confidence"] = 0.0f;
                }

                // Use the joint ID as the key in the JSON
                rawJointsJson[std::to_string(jointId)] = jointJson;
            }
            frameJson["rawJoints"] = rawJointsJson;

            // Add filtered joints if processed
            if (dataProcessed) {
                nlohmann::json filteredJointsJson = nlohmann::json::object();
                for (const auto& [jointId, jointPos] : frame.filteredJoints) {
                    nlohmann::json jointJson;
                    jointJson["x"] = jointPos.x;
                    jointJson["y"] = jointPos.y;
                    jointJson["z"] = jointPos.z;

                    // Add confidence if available
                    auto confIt = frame.confidences.find(jointId);
                    if (confIt != frame.confidences.end()) {
                        jointJson["confidence"] = confIt->second;
                    } else {
                        jointJson["confidence"] = 0.0f;
                    }

                    // Use the joint ID as the key in the JSON
                    filteredJointsJson[std::to_string(jointId)] = jointJson;
                }
                frameJson["filteredJoints"] = filteredJointsJson;
            }

            // Add frame to frames array
            outputJson["frames"].push_back(frameJson);
        }

        // Create directories if needed
        fs::create_directories(outputPath.parent_path());

        // Write JSON to file
        std::ofstream file(outputPath);
        if (!file.is_open()) {
            spdlog::error("Failed to open file for writing: {}", outputPath.string());
            return false;
        }

        file << outputJson.dump(2);  // Pretty print with indent of 2
        spdlog::info("Exported analysis data to: {}", outputPath.string());

        return true;
    } catch (const std::exception& e) {
        spdlog::error("Error exporting to JSON: {}", e.what());
        return false;
    }
}

cv::Mat JointProcessorAnalyzer::visualizeMotionTrails(int frameIndex, int trailLength, const cv::Mat& background) {
    // Validate frame index
    if (frameIndex < 0 || frameIndex >= static_cast<int>(frames.size())) {
        return cv::Mat();
    }

    // Create background or use provided one
    cv::Mat canvas;
    if (!background.empty()) {
        canvas = background.clone();
    } else {
        canvas = cv::Mat(720, 1280, CV_8UC3, cv::Scalar(40, 40, 40));
    }

    // Add title and frame info
    cv::putText(canvas, "Motion Trails Analysis", cv::Point(20, 30),
               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

    std::string frameInfo = "Frame: " + std::to_string(frameIndex) +
                           " (Trails: " + std::to_string(trailLength) + " frames)";
    cv::putText(canvas, frameInfo, cv::Point(20, 60),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 1);

    // Define key joints to track for trails
    std::vector<int> keyJoints = {
        JointType_Head,
        JointType_ShoulderLeft, JointType_ElbowLeft, JointType_WristLeft, JointType_HandLeft,
        JointType_ShoulderRight, JointType_ElbowRight, JointType_WristRight, JointType_HandRight,
        JointType_SpineMid,
        JointType_HipLeft, JointType_KneeLeft, JointType_AnkleLeft, JointType_FootLeft,
        JointType_HipRight, JointType_KneeRight, JointType_AnkleRight, JointType_FootRight
    };

    // Store trail points for each joint
    std::map<int, std::vector<cv::Point>> rawTrails;
    std::map<int, std::vector<cv::Point>> filteredTrails;

    // Calculate start frame for trails
    int startFrame = std::max(0, frameIndex - trailLength);

    // Project 3D coordinates to 2D for visualization
    for (int i = startFrame; i <= frameIndex; i++) {
        if (i >= static_cast<int>(frames.size())) break;

        // Calculate alpha value (transparency) based on temporal distance
        // More recent frames are more opaque
        float alpha = static_cast<float>(i - startFrame) / (frameIndex - startFrame);
        alpha = std::max(0.2f, alpha);  // Minimum 0.2 alpha for visibility

        // Process each key joint
        for (int jointId : keyJoints) {
            // Process raw joint
            auto rawIt = frames[i].rawJoints.find(jointId);
            if (rawIt != frames[i].rawJoints.end() &&
                !(rawIt->second.x == 0 && rawIt->second.y == 0 && rawIt->second.z == 0)) {

                // Project 3D point to 2D screen space (simple orthographic projection)
                cv::Point point(
                    static_cast<int>(rawIt->second.x * 300 + canvas.cols/2),
                    static_cast<int>(rawIt->second.y * 300 + canvas.rows/2)
                );

                // Add to trail if within bounds
                if (point.x >= 0 && point.x < canvas.cols &&
                    point.y >= 0 && point.y < canvas.rows) {
                    rawTrails[jointId].push_back(point);
                }
            }

            // Process filtered joint
            auto filteredIt = frames[i].filteredJoints.find(jointId);
            if (filteredIt != frames[i].filteredJoints.end() &&
                !(filteredIt->second.x == 0 && filteredIt->second.y == 0 && filteredIt->second.z == 0)) {

                // Project 3D point to 2D screen space
                cv::Point point(
                    static_cast<int>(filteredIt->second.x * 300 + canvas.cols/2),
                    static_cast<int>(filteredIt->second.y * 300 + canvas.rows/2)
                );

                // Add to trail if within bounds
                if (point.x >= 0 && point.x < canvas.cols &&
                    point.y >= 0 && point.y < canvas.rows) {
                    filteredTrails[jointId].push_back(point);
                }
            }
        }
    }

    // Draw the current skeleton for reference
    if (frameIndex < static_cast<int>(frames.size())) {
        drawSkeleton(canvas, frames[frameIndex].rawJoints, cv::Scalar(255, 0, 0), false);
        drawSkeleton(canvas, frames[frameIndex].filteredJoints, cv::Scalar(0, 255, 0), true);
    }

    // Draw trails for each joint
    for (const auto& trail : rawTrails) {
        if (trail.second.size() < 2) continue;

        // Draw raw trails in red
        for (size_t i = 1; i < trail.second.size(); i++) {
            // Calculate alpha based on temporal position
            float alpha = static_cast<float>(i) / trail.second.size();
            int thickness = 1 + (alpha > 0.7f ? 1 : 0);

            cv::line(canvas, trail.second[i-1], trail.second[i],
                    cv::Scalar(0, 0, 255), thickness);
        }
    }

    for (const auto& trail : filteredTrails) {
        if (trail.second.size() < 2) continue;

        // Draw filtered trails in green
        for (size_t i = 1; i < trail.second.size(); i++) {
            // Calculate alpha based on temporal position
            float alpha = static_cast<float>(i) / trail.second.size();
            int thickness = 1 + (alpha > 0.7f ? 1 : 0);

            cv::line(canvas, trail.second[i-1], trail.second[i],
                    cv::Scalar(0, 255, 0), thickness);
        }
    }

    // Add legend
    cv::putText(canvas, "Raw Motion Trail", cv::Point(20, canvas.rows - 40),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);
    cv::putText(canvas, "Filtered Motion Trail", cv::Point(20, canvas.rows - 20),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);

    return canvas;
}

cv::Mat JointProcessorAnalyzer::visualizeJointAngleConstraints(int frameIndex) {
    // Create visualization image
    cv::Mat result(720, 1280, CV_8UC3, cv::Scalar(40, 40, 40));

    if (frameIndex < 0 || frameIndex >= frames.size() || !dataProcessed) {
        cv::putText(result, "Invalid frame index: " + std::to_string(frameIndex),
                   cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1);
        return result;
    }

    // Define key joint angles to analyze
    struct AngleConstraint {
        int joint1;
        int joint2; // The central joint
        int joint3;
        float minAngle;
        float maxAngle;
        std::string name;
    };

    std::vector<AngleConstraint> constraints = {
        {JointType_ShoulderRight, JointType_ElbowRight, JointType_WristRight, 0.0f, 175.0f, "Right Elbow"},
        {JointType_ShoulderLeft, JointType_ElbowLeft, JointType_WristLeft, 0.0f, 175.0f, "Left Elbow"},
        {JointType_HipRight, JointType_KneeRight, JointType_AnkleRight, 0.0f, 180.0f, "Right Knee"},
        {JointType_HipLeft, JointType_KneeLeft, JointType_AnkleLeft, 0.0f, 180.0f, "Left Knee"},
        {JointType_SpineShoulder, JointType_ShoulderRight, JointType_ElbowRight, 0.0f, 180.0f, "Right Shoulder"},
        {JointType_SpineShoulder, JointType_ShoulderLeft, JointType_ElbowLeft, 0.0f, 180.0f, "Left Shoulder"},
        {JointType_Neck, JointType_Head, JointType_SpineShoulder, 150.0f, 180.0f, "Neck-Head"},
        {JointType_SpineMid, JointType_SpineShoulder, JointType_Neck, 150.0f, 180.0f, "Spine"}
    };

    // Title
    cv::putText(result, "Joint Angle Constraint Analysis - Frame " +
               std::to_string(frameIndex), cv::Point(20, 40),
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

    // Check constraints
    int y = 100;
    int violations = 0;
    int improvements = 0;

    for (const auto& constraint : constraints) {
        // Calculate angle for raw joints
        float rawAngle = calculateJointAngle(constraint.joint1, constraint.joint2,
                                          constraint.joint3, frames[frameIndex].rawJoints);

        // Calculate angle for filtered joints
        float filteredAngle = calculateJointAngle(constraint.joint1, constraint.joint2,
                                               constraint.joint3, frames[frameIndex].filteredJoints);

        // Check if constraints are violated
        bool rawViolated = rawAngle < constraint.minAngle || rawAngle > constraint.maxAngle;
        bool filteredViolated = filteredAngle < constraint.minAngle || filteredAngle > constraint.maxAngle;

        if (rawViolated && !filteredViolated) {
            improvements++;
        }
        if (filteredViolated) {
            violations++;
        }

        // Display the results
        std::string text = constraint.name + ": ";
        text += "Raw=" + std::to_string(static_cast<int>(rawAngle)) + "° ";
        text += rawViolated ? "(VIOLATION) " : "(OK) ";
        text += "Filtered=" + std::to_string(static_cast<int>(filteredAngle)) + "° ";
        text += filteredViolated ? "(VIOLATION)" : "(OK)";

        cv::Scalar textColor;
        if (rawViolated && !filteredViolated) {
            textColor = cv::Scalar(0, 255, 0); // Green - improved
        } else if (filteredViolated) {
            textColor = cv::Scalar(0, 0, 255); // Red - still violated
        } else {
            textColor = cv::Scalar(255, 255, 255); // White - no issue
        }

        putText(result, text, cv::Point(50, y), cv::FONT_HERSHEY_SIMPLEX,
                   0.6, textColor, 1);
        y += 40;

        // Draw angle gauge
        int gaugeX = 600;
        int gaugeWidth = 500;

        // Background
        cv::rectangle(result, cv::Point(gaugeX, y - 30),
                     cv::Point(gaugeX + gaugeWidth, y - 5),
                     cv::Scalar(100, 100, 100), -1);

        // Min/max markers
        float minPos = gaugeWidth * (constraint.minAngle / 180.0f);
        float maxPos = gaugeWidth * (constraint.maxAngle / 180.0f);

        line(result, cv::Point(gaugeX + minPos, y - 40),
                cv::Point(gaugeX + minPos, y), cv::Scalar(200, 200, 200), 1);
        line(result, cv::Point(gaugeX + maxPos, y - 40),
                cv::Point(gaugeX + maxPos, y), cv::Scalar(200, 200, 200), 1);

        // Raw angle marker
        float rawPos = gaugeWidth * (rawAngle / 180.0f);
        rawPos = std::min(std::max(0.0f, rawPos), static_cast<float>(gaugeWidth));
        circle(result, cv::Point(gaugeX + rawPos, y - 18), 10,
                  cv::Scalar(0, 0, 255), -1);

        // Filtered angle marker
        float filteredPos = gaugeWidth * (filteredAngle / 180.0f);
        filteredPos = std::min(std::max(0.0f, filteredPos), static_cast<float>(gaugeWidth));
        circle(result, cv::Point(gaugeX + filteredPos, y - 18), 7,
                  cv::Scalar(0, 255, 0), -1);

        y += 40;
    }

    // Summary
    y += 20;
    putText(result, "Summary:", cv::Point(50, y), cv::FONT_HERSHEY_SIMPLEX,
               0.7, cv::Scalar(255, 255, 255), 2);
    y += 40;

    putText(result, "Remaining violations: " + std::to_string(violations),
               cv::Point(50, y), cv::FONT_HERSHEY_SIMPLEX, 0.6,
               violations > 0 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255), 1);
    y += 30;

    putText(result, "Improvements: " + std::to_string(improvements),
               cv::Point(50, y), cv::FONT_HERSHEY_SIMPLEX, 0.6,
               improvements > 0 ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 255, 255), 1);
    y += 30;

    // Legend
    y = result.rows - 120;
    rectangle(result, cv::Point(50, y), cv::Point(300, y + 100),
                 cv::Scalar(40, 40, 40), -1);

    putText(result, "Legend:", cv::Point(60, y + 20), cv::FONT_HERSHEY_SIMPLEX,
               0.5, cv::Scalar(255, 255, 255), 1);

    circle(result, cv::Point(80, y + 50), 10, cv::Scalar(0, 0, 255), -1);
    putText(result, "Raw Joint Angle", cv::Point(100, y + 55),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    circle(result, cv::Point(80, y + 80), 7, cv::Scalar(0, 255, 0), -1);
    putText(result, "Filtered Joint Angle", cv::Point(100, y + 85),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    return result;
}

cv::Mat JointProcessorAnalyzer::visualizeStabilityMetrics(int frameStart, int frameEnd, int width, int height) {
    // Validate frame range
    frameStart = std::max(0, frameStart);
    frameEnd = std::min(frameEnd, static_cast<int>(frames.size()) - 1);

    if (frameStart > frameEnd || frames.empty()) {
        spdlog::error("Invalid frame range for stability metrics");
        return cv::Mat();
    }

    // Create canvas
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(40, 40, 40));

    // Extract raw and filtered joint data for the range
    std::vector<std::map<int, cv::Point3f>> rawJoints;
    std::vector<std::map<int, cv::Point3f>> filteredJoints;

    for (int i = frameStart; i <= frameEnd; i++) {
        rawJoints.push_back(frames[i].rawJoints);
        filteredJoints.push_back(frames[i].filteredJoints);
    }

    // Calculate stability metrics
    std::map<int, float> rawStability;
    std::map<int, float> filteredStability;

    jointProcessor.calculateStabilityMetrics(rawJoints, filteredJoints, rawStability, filteredStability);

    // Define key joints to visualize
    std::vector<int> keyJoints = {
        JointType_Head,
        JointType_ShoulderLeft, JointType_ShoulderRight,
        JointType_ElbowLeft, JointType_ElbowRight,
        JointType_WristLeft, JointType_WristRight,
        JointType_SpineMid,
        JointType_KneeLeft, JointType_KneeRight,
        JointType_AnkleLeft, JointType_AnkleRight
    };

    // Add title
    cv::putText(canvas, "Joint Stability Analysis (Frames " + std::to_string(frameStart) +
               " — " + std::to_string(frameEnd) + ")",
               cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);

    cv::putText(canvas, "Standard Deviation in mm (lower is better)",
               cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);

    // Draw the metric bars
    int barHeight = 20;
    int maxBarWidth = width - 300;
    int yPos = 100;
    float maxStability = 0;

    // Find maximum value for scaling
    for (int jointId : keyJoints) {
        if (rawStability.count(jointId)) {
            maxStability = std::max(maxStability, rawStability[jointId]);
        }
        if (filteredStability.count(jointId)) {
            maxStability = std::max(maxStability, filteredStability[jointId]);
        }
    }

    // Ensure we have a reasonable maximum
    maxStability = std::max(maxStability, 100.0f);

    // Calculate overall averages
    float avgRawStability = 0;
    float avgFilteredStability = 0;
    int jointCount = 0;

    for (int jointId : keyJoints) {
        if (rawStability.count(jointId) && filteredStability.count(jointId)) {
            avgRawStability += rawStability[jointId];
            avgFilteredStability += filteredStability[jointId];
            jointCount++;
        }
    }

    if (jointCount > 0) {
        avgRawStability /= jointCount;
        avgFilteredStability /= jointCount;
    }

    // Draw the joint stability bars
    for (int jointId : keyJoints) {
        if (rawStability.count(jointId) && filteredStability.count(jointId)) {
            std::string jointName = getJointName(jointId);

            // Draw joint name
            cv::putText(canvas, jointName, cv::Point(20, yPos + barHeight/2),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

            // Draw raw stability bar (red)
            int rawBarWidth = static_cast<int>(rawStability[jointId] / maxStability * maxBarWidth);
            cv::rectangle(canvas,
                         cv::Point(200, yPos),
                         cv::Point(200 + rawBarWidth, yPos + barHeight),
                         cv::Scalar(0, 0, 255),
                         -1);

            // Draw filtered stability bar (green)
            int filteredBarWidth = static_cast<int>(filteredStability[jointId] / maxStability * maxBarWidth);
            cv::rectangle(canvas,
                         cv::Point(200, yPos + barHeight + 5),
                         cv::Point(200 + filteredBarWidth, yPos + 2*barHeight + 5),
                         cv::Scalar(0, 255, 0),
                         -1);

            // Show values
            cv::putText(canvas,
                       std::to_string(static_cast<int>(rawStability[jointId])),
                       cv::Point(200 + rawBarWidth + 5, yPos + barHeight - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

            cv::putText(canvas,
                       std::to_string(static_cast<int>(filteredStability[jointId])),
                       cv::Point(200 + filteredBarWidth + 5, yPos + 2*barHeight),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);

            // Calculate improvement percentage
            float improvement = (rawStability[jointId] - filteredStability[jointId]) / rawStability[jointId] * 100.0f;

            std::string improvementText = (improvement >= 0) ?
                "+" + std::to_string(static_cast<int>(improvement)) + "%" :
                std::to_string(static_cast<int>(improvement)) + "%";

            cv::Scalar improvementColor = (improvement >= 0) ?
                cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

            cv::putText(canvas, improvementText,
                       cv::Point(width - 100, yPos + barHeight),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, improvementColor, 1);

            yPos += barHeight * 2 + 15;
        }
    }

    // Draw overall summary at bottom
    yPos = height - 60;
    cv::line(canvas, cv::Point(20, yPos - 10), cv::Point(width - 20, yPos - 10),
            cv::Scalar(150, 150, 150), 1);

    cv::putText(canvas, "OVERALL SUMMARY:", cv::Point(20, yPos + 20),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1);

    // Format numbers with up to 6 decimal places
    char rawText[64], filteredText[64];
    sprintf(rawText, "Average Raw Stability: %.6f mm", avgRawStability);
    sprintf(filteredText, "Average Filtered Stability: %.6f mm", avgFilteredStability);

    std::string rawStabilityText = rawText;
    std::string filteredStabilityText = filteredText;

    float overallImprovement = (avgRawStability - avgFilteredStability) / avgRawStability * 100.0f;
    std::string improvementText = "Overall Improvement: " +
                                std::to_string(static_cast<int>(overallImprovement)) + "%";

    cv::Scalar improvementColor = (overallImprovement >= 0) ?
        cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

    cv::putText(canvas, rawStabilityText, cv::Point(20, yPos + 50),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);

    cv::putText(canvas, filteredStabilityText, cv::Point(300, yPos + 50),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

    cv::putText(canvas, improvementText, cv::Point(600, yPos + 50),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, improvementColor, 1);

    return canvas;
}

cv::Mat JointProcessorAnalyzer::visualizeJointVelocity(int jointId, int frameStart, int frameEnd, int width, int height) {
    // Create a blank canvas
    cv::Mat result(height, width, CV_8UC3, cv::Scalar(40, 40, 40));

    if (frameStart < 0 || frameEnd >= frames.size() || frameStart > frameEnd || !dataProcessed) {
        cv::putText(result, "Invalid frame range: " + std::to_string(frameStart) + " - " +
                   std::to_string(frameEnd), cv::Point(20, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1);
        return result;
    }

    // Calculate velocity for the specified joint
    std::vector<float> velocities;
    std::vector<float> timestamps;

    for (int f = frameStart + 1; f <= frameEnd; f++) {
        auto prevIt = frames[f-1].filteredJoints.find(jointId);
        auto currIt = frames[f].filteredJoints.find(jointId);

        if (prevIt != frames[f-1].filteredJoints.end() &&
            currIt != frames[f].filteredJoints.end()) {

            // Calculate time interval
            float timeInterval = (frames[f].timestamp - frames[f-1].timestamp) / 1000000000.0f; // Convert to seconds

            // Calculate velocity
            float velocity = cv::norm(currIt->second - prevIt->second) / timeInterval;

            velocities.push_back(velocity);
            timestamps.push_back(timeInterval);
        }
    }

    // If no velocities calculated
    if (velocities.empty()) {
        cv::putText(result, "No velocity data available", cv::Point(20, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1);
        return result;
    }

    // Title
    std::string jointName = getJointName(jointId);
    cv::putText(result, "Velocity for " + jointName + " Joint",
               cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
               cv::Scalar(255, 255, 255), 2);

    // Find min and max velocities for scaling
    float minVelocity = *std::min_element(velocities.begin(), velocities.end());
    float maxVelocity = *std::max_element(velocities.begin(), velocities.end());

    // Plot velocities
    int plotHeight = height - 100;
    int plotWidth = width - 100;
    int plotX = 50;
    int plotY = height - 50;

    // Draw axes
    cv::line(result, cv::Point(plotX, plotY),
             cv::Point(plotX, plotY - plotHeight),
             cv::Scalar(200, 200, 200), 2);
    cv::line(result, cv::Point(plotX, plotY),
             cv::Point(plotX + plotWidth, plotY),
             cv::Scalar(200, 200, 200), 2);

    // Add axis labels
    cv::putText(result, "Time (s)",
               cv::Point(plotX + plotWidth/2, plotY + 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.5,
               cv::Scalar(200, 200, 200), 1);
    cv::putText(result, "Velocity (m/s)",
               cv::Point(10, plotY - plotHeight/2),
               cv::FONT_HERSHEY_SIMPLEX, 0.5,
               cv::Scalar(200, 200, 200), 1, cv::ROTATE_90_CLOCKWISE);

    // Plot velocity points and lines
    for (size_t i = 1; i < velocities.size(); i++) {
        float x1 = plotX + (timestamps[i-1] / (timestamps.back())) * plotWidth;
        float y1 = plotY - ((velocities[i-1] - minVelocity) / (maxVelocity - minVelocity)) * plotHeight;

        float x2 = plotX + (timestamps[i] / (timestamps.back())) * plotWidth;
        float y2 = plotY - ((velocities[i] - minVelocity) / (maxVelocity - minVelocity)) * plotHeight;

        cv::line(result,
                 cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                 cv::Point(static_cast<int>(x2), static_cast<int>(y2)),
                 cv::Scalar(0, 255, 0), 2);
    }

    // Add velocity range info
    std::stringstream ss;
    ss << "Velocity Range: " << std::fixed << std::setprecision(2)
       << minVelocity << " - " << maxVelocity << " m/s";
    cv::putText(result, ss.str(),
               cv::Point(20, height - 20),
               cv::FONT_HERSHEY_SIMPLEX, 0.5,
               cv::Scalar(200, 200, 200), 1);

    return result;
}

bool JointProcessorAnalyzer::generateAnalysisVideo(const std::filesystem::path& outputPath,
                                                 bool includeSideBySide,
                                                 bool includeMotionTrails,
                                                 bool includeVelocity) {
    if (frames.empty()) {
        spdlog::error("No frames to generate video from");
        return false;
    }

    // Calculate actual frame rate from timestamps
    double actualFps = 30.0;  // Default fallback
    if (frames.size() > 1) {
        uint64_t firstTs = frames.front().timestamp;
        uint64_t lastTs = frames.back().timestamp;
        double durationSec = (lastTs - firstTs) / 1000000000.0;  // Convert nanoseconds to seconds
        if (durationSec > 0) {
            actualFps = frames.size() / durationSec;
            spdlog::info("Calculated actual frame rate: {:.2f} FPS", actualFps);
        }
    }

    // Prepare video writer
    cv::Size frameSize(1280, 720);
    cv::VideoWriter videoWriter;

    // Try different codecs
    std::vector<std::pair<std::string, int>> codecs = {
        {"XVID", cv::VideoWriter::fourcc('X', 'V', 'I', 'D')},
        {"MJPG", cv::VideoWriter::fourcc('M', 'J', 'P', 'G')},
        {"MP4V", cv::VideoWriter::fourcc('M', 'P', '4', 'V')}
    };

    bool writerInitialized = false;
    for (const auto& codec : codecs) {
        spdlog::info("Trying codec: {}", codec.first);
        if (videoWriter.open(outputPath.string(), codec.second, actualFps, frameSize)) {
            writerInitialized = true;
            spdlog::info("Successfully opened video writer with codec: {}", codec.first);
            break;
        }
    }

    if (!writerInitialized) {
        spdlog::error("Failed to create video writer");
        return false;
    }

    // Process each frame
    int totalFrames = static_cast<int>(frames.size());
    for (int i = 0; i < totalFrames; i++) {
        // Create a blank canvas
        cv::Mat outputFrame = cv::Mat(frameSize, CV_8UC3, cv::Scalar(40, 40, 40));

        // Add frame number and timestamp
        std::string frameText = "Frame: " + std::to_string(i) + "/" + std::to_string(totalFrames-1);
        cv::putText(outputFrame, frameText, cv::Point(20, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 1);

        // Add side-by-side comparison
        if (includeSideBySide) {
            cv::Mat comparison = visualizeSideBySide(i);
            if (!comparison.empty()) {
                // Resize to fit in the top section of the output frame
                cv::Mat resizedComp;
                cv::resize(comparison, resizedComp, cv::Size(frameSize.width, frameSize.height/2-30));
                resizedComp.copyTo(outputFrame(cv::Rect(0, 50, resizedComp.cols, resizedComp.rows)));
            }
        }

        // Add motion trails
        if (includeMotionTrails && i > 0) {
            cv::Mat trails = visualizeMotionTrails(i, 20);
            if (!trails.empty()) {
                // Resize to fit in the bottom-left section
                cv::Mat resizedTrails;
                int trailsHeight = frameSize.height/2-30;
                int trailsWidth = frameSize.width/2;
                cv::resize(trails, resizedTrails, cv::Size(trailsWidth, trailsHeight));
                resizedTrails.copyTo(outputFrame(cv::Rect(0, frameSize.height/2+20, trailsWidth, trailsHeight)));
            }
        }

        // Add velocity graph for selected joint
        if (includeVelocity) {
            // Select a key joint to track velocity (e.g., spine mid as center of mass)
            int velocityJointId = JointType_SpineMid;

            // Create velocity visualization for a window around the current frame
            int startFrame = std::max(0, i - 30);
            int endFrame = std::min(totalFrames - 1, i + 30);

            cv::Mat velocityGraph = visualizeJointVelocity(velocityJointId, startFrame, endFrame,
                                                         frameSize.width/2, frameSize.height/2-50);

            if (!velocityGraph.empty()) {
                velocityGraph.copyTo(outputFrame(cv::Rect(frameSize.width/2, frameSize.height/2+20,
                                                        velocityGraph.cols, velocityGraph.rows)));
            }
        }

        // Write frame to video
        videoWriter.write(outputFrame);

        // Log progress
        if (i % 50 == 0 || i == totalFrames - 1) {
            spdlog::info("Video generation progress: {:.1f}% ({}/{})",
                       (100.0 * i) / totalFrames, i + 1, totalFrames);
        }
    }

    videoWriter.release();
    spdlog::info("Analysis video saved to {}", outputPath.string());
    return true;
}

