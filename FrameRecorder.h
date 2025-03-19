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
#include <future>
#include <opencv2/opencv.hpp>

class IOThreadPool;
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
        bool useVideoCompression = false;
        int targetFps = 30;
        bool limitFrameRate = false;
        int processEveryNFrames = 1;
        std::string outputDirectory = "recordings";
        bool saveOriginalFrames = true;
        bool generateVideoOnRecording = false;
        bool useJpegForColorFrames = true;
        int jpegQuality = 90;
    };

    // Frame data structure
    struct FrameData {
        cv::Mat colorImage;
        cv::Mat depthImage;
        std::chrono::system_clock::time_point timestamp;
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
    void createRecordingVideo(int totalFrames, double actualFps, std::chrono::duration<double> duration);

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

    // Thread pool for I/O operations
    std::unique_ptr<IOThreadPool> ioPool;

    // Pending operation tracking
    std::map<uint64_t, std::future<void>> pendingOperations;
    std::mutex pendingOpsMutex;
    size_t maxConcurrentOperations;

    // Helper methods
    bool saveFrameToDisk(const FrameData& frame, int frameIndex, uint64_t operationId);
    void cleanupCompletedOperations();
    void waitForOperationsIfNeeded();
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