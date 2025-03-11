#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <functional>
#include <filesystem>
#include <future>

#include <opencv2/opencv.hpp>
#include <Kinect.h>
#include <nlohmann/json.hpp>

class OpenPoseInterfaceImpl;

/**
 * @brief Modern wrapper for OpenPose integration
 *
 * This class provides a clean, modern C++ interface for working with OpenPose,
 * with proper resource management and error handling.
 */
class OpenPoseInterface {
public:
    // Configuration settings
    struct Configuration {
        std::filesystem::path openPoseExePath = "bin/OpenPoseDemo.exe";
        int netResolution = 368;
        bool useMaximumAccuracy = false;
        int keypointConfidenceThreshold = 40;
        int batchSize = 20;  // Process frames in batches for better performance
        bool performanceMode = false;
        std::filesystem::path tempDirectory = "temp_frames";
        std::filesystem::path outputJsonDirectory = "output_json";
    };

    // 3D keypoint data
    struct Keypoint3D {
        // 2D coordinates in image space (for visualization)
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;  // Depth value

        // 3D coordinates in world space
        float world_x = 0.0f;
        float world_y = 0.0f;
        float world_z = 0.0f;

        float confidence = 0.0f;
    };

    // 3D person data
    struct Person3D {
        std::vector<Keypoint3D> keypoints;
    };

    // Processing progress callback
    using ProgressCallback = std::function<void(int current, int total)>;

    // Results of processing a recording
    struct ProcessingResult {
        int framesProcessed = 0;
        int peopleDetected = 0;
        std::filesystem::path outputDirectory;
        double processingTimeSeconds = 0.0;
    };

    // Constructor and destructor
    explicit OpenPoseInterface(Configuration config = {});
    ~OpenPoseInterface();

    // No copy
    OpenPoseInterface(const OpenPoseInterface&) = delete;
    OpenPoseInterface& operator=(const OpenPoseInterface&) = delete;

    // Allow move
    OpenPoseInterface(OpenPoseInterface&&) noexcept;
    OpenPoseInterface& operator=(OpenPoseInterface&&) noexcept;

    // Initialize the interface
    bool initialize();
    bool isInitialized() const noexcept;

    // Configuration
    void setConfiguration(const Configuration& config);
    const Configuration& getConfiguration() const;

    // Process a single frame
    std::optional<std::vector<Person3D>> processFrame(
        const cv::Mat& colorImage,
        const cv::Mat& depthImage,
        ICoordinateMapper* coordinateMapper
    );

    // Process a recording directory asynchronously
    std::future<ProcessingResult> processRecordingAsync(
        const std::filesystem::path& recordingDir,
        ICoordinateMapper* coordinateMapper,
        const std::filesystem::path& outputDir,
        int numThreads = 4,
        ProgressCallback progressCallback = nullptr,
        bool cleanupTempFiles = false
    );

    // Process a recording directory (synchronous version)
    ProcessingResult processRecording(
        const std::filesystem::path& recordingDir,
        ICoordinateMapper* coordinateMapper,
        const std::filesystem::path& outputDir,
        int numThreads = 4,
        ProgressCallback progressCallback = nullptr,
        bool cleanupTempFiles = false
    );

    // Static utility methods
    static bool save3DSkeletonToJson(
        const std::vector<Person3D>& people,
        const std::filesystem::path& outputPath
    );

    static cv::Mat visualize3DSkeleton(
        const cv::Mat& image,
        const std::vector<Person3D>& people
    );

private:
    // PIMPL idiom
    std::unique_ptr<OpenPoseInterfaceImpl> pImpl;
};