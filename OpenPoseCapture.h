#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Kinect.h>
#include <nlohmann/json.hpp>

// For convenience when working with JSON
using json = nlohmann::json;

// Structure to hold 3D keypoint data
struct Keypoint3D {
    float x;
    float y;
    float z;
    float confidence;
};

// Structure to hold OpenPose person data with 3D information
struct Person3D {
    std::vector<Keypoint3D> keypoints;
};

class OpenPoseCapture {
private:
    // Paths
    std::string openPoseExePath;
    std::string tempImageDir;
    std::string outputJsonDir;

    // Processing settings
    int netResolution;
    bool useMaximumAccuracy;
    int keypointConfidenceThreshold;

    // OpenPose model mapping to Kinect joints
    std::unordered_map<int, JointType> openposeToKinectJointMap;

    // Temp storage
    cv::Mat colorMat;

    // Methods
    void initJointMapping();
    bool runOpenPoseOnImage(const std::string& imagePath, const std::string& outputDir) const;

    static json readOpenPoseJson(const std::string& jsonPath);

    static std::string getLastJsonFile(const std::string& directory);

public:
    explicit OpenPoseCapture(std::string  openPosePath = "bin\\OpenPoseDemo.exe",
                   std::string  tempDir = "temp_frames",
                   std::string  jsonDir = "output_json");
    ~OpenPoseCapture();

    // Initialize settings
    void setNetResolution(int resolution) { netResolution = resolution; }
    void setMaximumAccuracy(bool enabled) { useMaximumAccuracy = enabled; }
    void setKeypointConfidenceThreshold(int threshold) { keypointConfidenceThreshold = threshold; }

    // Core processing methods
    bool initialize();
    bool processFrame(const cv::Mat& colorImage, const cv::Mat& depthImage, ICoordinateMapper* coordinateMapper, std::vector<Person3D>& detectedPeople) const;

    // Save results
    static bool save3DSkeletonToJson(const std::vector<Person3D>& people, const std::string& outputPath);

    // Visualization
    static void visualize3DSkeleton(cv::Mat& image, const std::vector<Person3D>& people);
};