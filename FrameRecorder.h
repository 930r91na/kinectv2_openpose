#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Kinect.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <filesystem>
#include <queue>
#include <condition_variable>
#include <fstream>

struct FramePair {
    cv::Mat colorImage;
    cv::Mat depthImage;
    UINT64 timestamp;
};

class FrameRecorder {
private:
    std::queue<FramePair> frameQueue;     // Buffer queue for frames
    std::string outputDirectory;
    std::atomic<bool> isRecording{false};
    std::mutex framesMutex;
    std::thread writerThread;
    std::atomic<bool> shouldWrite{false};
    std::atomic<bool> isProcessing{false};
    std::condition_variable processingCv;

    // Stream handling
    cv::VideoWriter colorVideoWriter;
    std::ofstream metadataStream;
    std::string currentRecordingPath;
    std::atomic<int> frameCounter{0}; // Make atomic for thread safety

    // Timestamp tracking for FPS calculation
    std::chrono::high_resolution_clock::time_point recordingStartTime;
    std::chrono::high_resolution_clock::time_point lastFrameTime;
    std::vector<uint64_t> frameTimestamps;

    // Configuration
    const size_t MAX_QUEUE_SIZE = 30;    // Buffer ~1 second at 30fps
    int processingInterval = 1;          // Process every N frames
    bool useVideoCompression = true;     // Use video compression for color frames
    int targetFps = 30;                  // Target recording FPS
    bool frameLimitingEnabled = false;    // Limit frame rate to target FPS

    void processFrameQueue();

public:
    explicit FrameRecorder(std::string outputDir = "recorded_frames");
    ~FrameRecorder();

    void startRecording();
    void stopRecording();
    void addFrame(const cv::Mat& colorImg, const cv::Mat& depthImg);
    bool saveFramesToDisk();
    size_t getFrameCount() const { return frameCounter; }

    // Configuration options
    void setProcessingInterval(int interval) { processingInterval = interval; }
    void setUseVideoCompression(bool use) { useVideoCompression = use; }
    void setTargetFPS(int fps);
    void setFrameLimiting(bool enabled);

    int getTargetFPS() const { return targetFps; }
    bool isFrameLimitingEnabled() const { return frameLimitingEnabled; }

    static bool loadRecordedFrames(const std::string& directory, std::vector<FramePair>& outFrames);
};