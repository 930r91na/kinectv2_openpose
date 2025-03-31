#pragma once

#include "JointProcessor.h"
#include <filesystem>
#include <vector>
#include <map>
#include <string>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

class JointProcessorAnalyzer {
public:
    struct FrameData {
        int frameIndex;
        std::map<int, cv::Point3f> rawJoints;
        std::map<int, float> confidences;
        std::map<int, cv::Point3f> filteredJoints;
        uint64_t timestamp;
    };

    struct RecordingMetadata {
        std::string recordingId;
        std::string startDateTime;
        int totalFrames = 0;
        double durationSeconds = 0.0;
        float averageFps = 0.0f;
        std::map<std::string, std::string> additionalInfo;
    };

    JointProcessorAnalyzer();
    ~JointProcessorAnalyzer();

    // Load recording data
    bool loadRecording(const std::filesystem::path& recordingPath, const std::filesystem::path& processedPath);

    int mapOpenPoseToKinectJoint(int openposeIndex) {
        // OpenPose BODY_25 keypoints to Kinect joint types mapping
        static const std::map<int, int> openposeToKinectMap = {
            {0, JointType_Head},            // Nose -> Head
            {1, JointType_Neck},            // Neck -> Neck
            {2, JointType_ShoulderRight},   // RShoulder -> ShoulderRight
            {3, JointType_ElbowRight},      // RElbow -> ElbowRight
            {4, JointType_WristRight},      // RWrist -> WristRight
            {5, JointType_ShoulderLeft},    // LShoulder -> ShoulderLeft
            {6, JointType_ElbowLeft},       // LElbow -> ElbowLeft
            {7, JointType_WristLeft},       // LWrist -> WristLeft
            {8, JointType_SpineBase},       // MidHip -> SpineBase
            {9, JointType_HipRight},        // RHip -> HipRight
            {10, JointType_KneeRight},      // RKnee -> KneeRight
            {11, JointType_AnkleRight},     // RAnkle -> AnkleRight
            {12, JointType_HipLeft},        // LHip -> HipLeft
            {13, JointType_KneeLeft},       // LKnee -> KneeLeft
            {14, JointType_AnkleLeft},      // LAnkle -> AnkleLeft
            {15, JointType_Head},           // REye -> Head (approximate)
            {16, JointType_Head},           // LEye -> Head (approximate)
            {17, JointType_Head},           // REar -> Head (approximate)
            {18, JointType_Head},           // LEar -> Head (approximate)
            // Additional mappings for Body_25 points (feet)
            {19, JointType_FootLeft},       // LBigToe -> FootLeft
            {20, JointType_FootLeft},       // LSmallToe -> FootLeft
            {21, JointType_FootLeft},       // LHeel -> FootLeft
            {22, JointType_FootRight},      // RBigToe -> FootRight
            {23, JointType_FootRight},      // RSmallToe -> FootRight
            {24, JointType_FootRight}       // RHeel -> FootRight
        };

        auto it = openposeToKinectMap.find(openposeIndex);
        if (it != openposeToKinectMap.end()) {
            return it->second;
        }
        return -1; // Invalid mapping
    }

    // Process data through JointProcessor
    bool processData();

    // Analysis and visualization methods
    cv::Mat visualizeSideBySide(int frameIndex, const cv::Mat& background = cv::Mat());
    cv::Mat visualizeMotionTrails(int frameIndex, int trailLength = 20, const cv::Mat& background = cv::Mat());
    cv::Mat visualizeJointVelocity(int jointId, int frameStart, int frameEnd, int width = 800, int height = 400);
    cv::Mat visualizeJointAngleConstraints(int frameIndex);
    cv::Mat visualizeStabilityMetrics(int frameStart, int frameEnd, int width = 800, int height = 600);

    // Export processed data
    bool exportToJson(const std::filesystem::path& outputPath);

    // Generate a video from the analysis
    bool generateAnalysisVideo(const std::filesystem::path& outputPath, bool includeSideBySide = true,
                              bool includeMotionTrails = true, bool includeVelocity = true);

    // Get frame count
    int getFrameCount() const { return static_cast<int>(frames.size()); }

private:
    // Helper methods
    bool loadDepthData(int frameIndex, cv::Mat& depthMat);
    bool loadJsonData(int frameIndex, nlohmann::json& jsonData);
    bool loadMetadataFromRecording();
    cv::Point3f calculateJointVelocity(const cv::Point3f& pos1, const cv::Point3f& pos2, float timeInterval);
    float calculateJointAngle(int jointId1, int jointId2, int jointId3, const std::map<int, cv::Point3f>& joints);
    static void drawSkeleton(cv::Mat& image, const std::map<int, cv::Point3f>& joints, cv::Scalar color, bool isFiltered = false);
    std::string getJointName(int jointId);

    // Data members
    std::filesystem::path recordingPath;
    std::filesystem::path processedPath;
    std::filesystem::path depthRawPath;
    std::filesystem::path jsonPath;
    std::filesystem::path processingTempPath;

    std::vector<FrameData> frames;
    std::vector<uint64_t> frameTimestamps;
    RecordingMetadata metadata;

    JointProcessor jointProcessor;
    bool dataLoaded;
    bool dataProcessed;
    bool metadataLoaded;
};