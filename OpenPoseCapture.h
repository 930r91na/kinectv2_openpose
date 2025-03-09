#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Kinect.h>
#include <nlohmann/json.hpp>
#include <mutex>
#include <atomic>
#include <thread>
#include <queue>
#include <condition_variable>

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

// Processing job data structure
struct ProcessingJob {
    std::string colorPath;
    std::string depthPath;
    int frameIndex;
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
    int batchSize;
    bool performanceMode = false;  // New flag for faster processing mode

    // OpenPose model mapping to Kinect joints
    std::unordered_map<int, JointType> openposeToKinectJointMap;

    // Multi-threading for batch processing
    std::mutex jobMutex;
    std::condition_variable jobCV;
    std::queue<ProcessingJob> jobQueue;
    std::vector<std::thread> workerThreads;
    std::atomic<bool> isProcessing{false};
    std::atomic<int> jobsCompleted{0};
    std::atomic<int> totalJobs{0};

    // Coordinate mapper shared between threads
    ICoordinateMapper* m_pCoordinateMapper;
    std::mutex coordinateMapperMutex;

    // Methods
    void initJointMapping();
    bool runOpenPoseOnImage(const std::string& imagePath, const std::string& outputDir) const;
    static json readOpenPoseJson(const std::string& jsonPath);
    static std::string getLastJsonFile(const std::string& directory);

    // Worker thread for processing batches
    void processingWorker(const std::string& outputDir);

    // Process 3D lifting without creating temporary files
    bool process3DLifting(const cv::Mat& colorImage,
                         const cv::Mat& depthImage,
                         ICoordinateMapper* coordinateMapper,
                         const json& openposeData,
                         std::vector<Person3D>& detectedPeople) const;

    // Load depth data
    static cv::Mat loadRawDepthData(const std::string& depthPath);

public:
    explicit OpenPoseCapture(std::string  openPosePath = "bin\\OpenPoseDemo.exe",
                           std::string  tempDir = "temp_frames",
                           std::string  jsonDir = "output_json");
    ~OpenPoseCapture();

    // Initialize settings
    void setNetResolution(int resolution) { netResolution = resolution; }
    void setMaximumAccuracy(bool enabled) { useMaximumAccuracy = enabled; }
    void setKeypointConfidenceThreshold(int threshold) { keypointConfidenceThreshold = threshold; }
    void setBatchSize(int size) { batchSize = size; }
    void setPerformanceMode(bool enabled) { performanceMode = enabled; }

    // Core processing methods
    bool initialize();
    bool processFrame(const cv::Mat& colorImage, const cv::Mat& depthImage,
                     ICoordinateMapper* coordinateMapper,
                     std::vector<Person3D>& detectedPeople) const;

    // Batch processing methods
    bool processBatch(const std::vector<std::string>& colorImagePaths,
                     const std::vector<std::string>& depthRawPaths,
                     ICoordinateMapper* coordinateMapper,
                     const std::string& outputDir) const;

    bool processRecordingDirectory(const std::string& recordingDir,
                                 ICoordinateMapper* coordinateMapper,
                                 const std::string& outputDir,
                                 int numThreads = 4);

    // Save results
    static bool save3DSkeletonToJson(const std::vector<Person3D>& people, const std::string& outputPath);

    // Visualization
    static void visualize3DSkeleton(cv::Mat& image, const std::vector<Person3D>& people);
};