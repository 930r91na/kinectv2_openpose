#pragma once

#include <string>
#include <filesystem>
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <optional>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>

/**
 * @brief Modern frame recorder for Kinect data
 *
 * This class handles recording and storage of color and depth frames
 * from the Kinect, using modern C++ features and thread-safe operations.
 */
class FrameRecorder {
public:
    // Configuration options
    struct RecordingOptions {
        bool useVideoCompression = true;
        int targetFps = 30;
        bool limitFrameRate = false;
        int processEveryNFrames = 1;
        std::string outputDirectory = "recordings";
        bool saveOriginalFrames = true;
    };

    // Frame data structure
    struct FrameData {
        cv::Mat colorImage;
        cv::Mat depthImage;
        std::chrono::system_clock::time_point timestamp;
    };

    // Statistics from a recording session
    struct RecordingStats {
        int totalFrames = 0;
        int colorFramesSaved = 0;
        int depthFramesSaved = 0;
        double recordingDurationSeconds = 0.0;
        double averageFps = 0.0;
        std::string outputPath;
    };

    // Constructor and destructor
    explicit FrameRecorder(RecordingOptions options = {});
    ~FrameRecorder();

    // Disable copying
    FrameRecorder(const FrameRecorder&) = delete;
    FrameRecorder& operator=(const FrameRecorder&) = delete;

    // Enable moving
    FrameRecorder(FrameRecorder&&) noexcept;
    FrameRecorder& operator=(FrameRecorder&&) noexcept;

    // Recording control
    bool startRecording(const std::string& sessionId = "");
    bool stopRecording();
    bool isRecording() const noexcept;

    // Frame submission - returns true if frame was accepted
    bool addFrame(const cv::Mat& colorFrame, const cv::Mat& depthFrame);

    // Status and statistics
    int getFrameCount() const noexcept;

    // Options
    void setRecordingOptions(const RecordingOptions& newOptions);
    const RecordingOptions& getRecordingOptions() const noexcept;

    // Load recorded frames
    static std::vector<FrameData> loadRecordedFrames(const std::string& directory);

    // Cleanup temporary files after processing
    void cleanupProcessingTemp(const std::string& sessionId);

private:
    // Implementation details
    struct RecordingSession {
        std::filesystem::path outputPath;
        std::chrono::system_clock::time_point startTime;
        std::chrono::system_clock::time_point lastFrameTime;
        std::atomic<int> frameCounter{0};
        std::vector<int64_t> frameTimestamps;
        std::ofstream metadataStream;
        cv::VideoWriter colorVideoWriter;
    };

    // Processing thread function
    void processFrameQueue();

    // Helper methods
    bool saveFrameToDisk(const FrameData& frame, int frameIndex);
    std::string generateUniqueSessionId(const std::string& sessionId) const;

    // New helper methods
    cv::Mat getFrameFromSource(int frameIndex) const;
    cv::Mat createFrameWithOverlay(const cv::Mat& originalFrame, int frameIndex);

    // Member variables
    RecordingOptions options;
    std::unique_ptr<RecordingSession> currentSession;
    std::atomic<bool> isRecordingActive{false};

    // Thread synchronization
    std::queue<FrameData> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCondition;
    std::thread processingThread;
    std::atomic<bool> shouldProcessFrames{false};

    // Constants
    static constexpr size_t MAX_QUEUE_SIZE = 60;
};