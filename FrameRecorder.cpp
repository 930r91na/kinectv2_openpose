#include "FrameRecorder.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <future>
#include <condition_variable>
#include <queue>

namespace fs = std::filesystem;

// I/O Thread Pool for controlled parallelism
class IOThreadPool {
public:
    explicit IOThreadPool(size_t numThreads) : stop(false) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this] {
                            return !tasks.empty() || stop;
                        });

                        if (stop && tasks.empty()) {
                            return;
                        }

                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    std::future<void> enqueue(F&& f) {
        auto task = std::make_shared<std::packaged_task<void()>>(std::forward<F>(f));
        std::future<void> result = task->get_future();

        {
            std::unique_lock<std::mutex> lock(queueMutex);
            if (stop) {
                throw std::runtime_error("ThreadPool is stopped");
            }
            tasks.emplace([task]() { (*task)(); });
        }

        condition.notify_one();
        return result;
    }

    size_t pendingTasks() const {
        std::unique_lock<std::mutex> lock(queueMutex);
        return tasks.size();
    }

    ~IOThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    mutable std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};

FrameRecorder::FrameRecorder(RecordingOptions opts)
    : options(std::move(opts)),
      ioPool(std::make_unique<IOThreadPool>(std::thread::hardware_concurrency() - 4)), // cpu has 10
      maxConcurrentOperations(16) { // Limit concurrent operations
    spdlog::info("Number of threads in pool: {}", std::thread::hardware_concurrency() - 4);
    if (!fs::exists(options.outputDirectory)) {
        try {
            fs::create_directories(options.outputDirectory);
            spdlog::info("Created output directory: {}", options.outputDirectory);
        } catch (const fs::filesystem_error& e) {
            spdlog::error("Failed to create output directory: {}", e.what());
        }
    }
}

FrameRecorder::~FrameRecorder() {
    // Make sure recording is stopped
    if (isRecordingActive) {
        try {
            stopRecording();
        } catch (const std::exception& e) {
            spdlog::error("Exception during stopRecording in destructor: {}", e.what());
        } catch (...) {
            spdlog::error("Unknown exception during stopRecording in destructor");
        }
    }

    // Make sure processing thread is stopped
    shouldProcessFrames = false;
    queueCondition.notify_all();

    if (processingThread.joinable()) {
        try {
            processingThread.join();
        } catch (const std::exception& e) {
            spdlog::error("Exception joining processing thread: {}", e.what());
        }
    }

    // Handle any remaining pending operations
    try {
        std::lock_guard<std::mutex> lock(pendingOpsMutex);
        pendingOperations.clear();
    } catch (...) {
        spdlog::error("Exception clearing pending operations in destructor");
    }
}

FrameRecorder::FrameRecorder(FrameRecorder&& other) noexcept
    : options(std::move(other.options)),
      currentSession(std::move(other.currentSession)),
      isRecordingActive(other.isRecordingActive.load()),
      shouldProcessFrames(other.shouldProcessFrames.load()),
      ioPool(std::move(other.ioPool)),
      maxConcurrentOperations(other.maxConcurrentOperations) {

    // Move queued frames
    {
        std::lock_guard<std::mutex> lock(other.queueMutex);
        frameQueue = std::move(other.frameQueue);
    }

    // Move pending operations
    {
        std::lock_guard<std::mutex> lock(other.pendingOpsMutex);
        pendingOperations = std::move(other.pendingOperations);
    }

    // Handle processing thread
    other.shouldProcessFrames = false;
    other.queueCondition.notify_all();

    if (other.processingThread.joinable()) {
        other.processingThread.join();
    }

    // Start a new processing thread if needed
    if (shouldProcessFrames) {
        processingThread = std::thread(&FrameRecorder::processFrameQueue, this);
    }

    // Reset other's state
    other.isRecordingActive = false;
}

FrameRecorder& FrameRecorder::operator=(FrameRecorder&& other) noexcept {
    if (this != &other) {
        // Stop current recording and processing
        if (isRecordingActive) {
            stopRecording();
        }

        shouldProcessFrames = false;
        queueCondition.notify_all();

        if (processingThread.joinable()) {
            processingThread.join();
        }

        // Wait for all pending file operations to complete with timeout and exception handling
        {
            std::lock_guard<std::mutex> lock(pendingOpsMutex);
            spdlog::info("Waiting for {} pending file operations to complete", pendingOperations.size());

            for (auto it = pendingOperations.begin(); it != pendingOperations.end(); ) {
                try {
                    // First check if already done without waiting
                    if (it->second.valid() &&
                        it->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                        it = pendingOperations.erase(it);
                        continue;
                        }

                    // Not done - wait with timeout for safety
                    if (it->second.valid()) {
                        auto status = it->second.wait_for(std::chrono::seconds(1));
                        if (status == std::future_status::ready || status == std::future_status::deferred) {
                            it = pendingOperations.erase(it);
                            continue;
                        } else {
                            // Timed out - log and move on
                            spdlog::warn("Operation {} timed out, continuing shutdown", it->first);
                            ++it;
                        }
                    } else {
                        // Invalid future - remove and continue
                        it = pendingOperations.erase(it);
                    }
                } catch (const std::exception& e) {
                    // Handle any exceptions from futures
                    spdlog::error("Exception while waiting for operation {}: {}", it->first, e.what());
                    it = pendingOperations.erase(it);
                } catch (...) {
                    // Catch all other exceptions
                    spdlog::error("Unknown exception while waiting for operation {}", it->first);
                    it = pendingOperations.erase(it);
                }
            }

            pendingOperations.clear();
            spdlog::info("All pending operations handled");
        }

        // Move data from other
        options = std::move(other.options);
        currentSession = std::move(other.currentSession);
        isRecordingActive = other.isRecordingActive.load();
        shouldProcessFrames = other.shouldProcessFrames.load();
        ioPool = std::move(other.ioPool);
        maxConcurrentOperations = other.maxConcurrentOperations;

        // Move queued frames
        {
            std::lock_guard<std::mutex> lock(other.queueMutex);
            frameQueue = std::move(other.frameQueue);
        }

        // Move pending operations
        {
            std::lock_guard<std::mutex> lock(other.pendingOpsMutex);
            pendingOperations = std::move(other.pendingOperations);
        }

        // Stop other's processing thread
        other.shouldProcessFrames = false;
        other.queueCondition.notify_all();

        if (other.processingThread.joinable()) {
            other.processingThread.join();
        }

        // Start new processing thread if needed
        if (shouldProcessFrames) {
            processingThread = std::thread(&FrameRecorder::processFrameQueue, this);
        }

        // Reset other's state
        other.isRecordingActive = false;
    }

    return *this;
}

bool FrameRecorder::isRecording() const noexcept {
    return isRecordingActive;
}

bool FrameRecorder::addFrame(const cv::Mat& colorFrame, const cv::Mat& depthFrame) {
    if (!isRecordingActive || !currentSession) {
        return false;
    }

    // Frame rate limiting
    if (options.limitFrameRate && options.targetFps > 0) {
        auto currentTime = std::chrono::system_clock::now();
        std::chrono::duration<double, std::milli> timeSinceLastFrame =
            currentTime - currentSession->lastFrameTime;

        double minFrameIntervalMs = 1000.0 / options.targetFps;

        if (timeSinceLastFrame.count() < minFrameIntervalMs) {
            // Skip this frame to maintain target FPS
            return false;
        }

        // Update last frame time
        currentSession->lastFrameTime = currentTime;
    }

    // Check if frames are valid
    if (colorFrame.empty() || depthFrame.empty()) {
        spdlog::warn("Skipping empty frame");
        return false;
    }

    // Create a new frame data object - use cv::Mat's reference counting
    FrameData frameData;
    frameData.colorImage = colorFrame; // Uses reference counting
    frameData.depthImage = depthFrame; // Uses reference counting
    frameData.timestamp = std::chrono::system_clock::now();

    // Store timestamp
    auto timestampNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        frameData.timestamp.time_since_epoch()).count();
    currentSession->frameTimestamps.push_back(timestampNs);

    // More aggressive frame queue management
    const int OPTIMAL_QUEUE_SIZE = 20; // Balanced for throughput vs memory usage

    // Add to processing queue
    {
        std::unique_lock<std::mutex> lock(queueMutex);

        // Wait if queue is full to prevent excessive memory usage
        queueCondition.wait(lock, [this, OPTIMAL_QUEUE_SIZE] {
            return frameQueue.size() < OPTIMAL_QUEUE_SIZE || !isRecordingActive;
        });

        if (!isRecordingActive) {
            return false;
        }

        frameQueue.push(std::move(frameData));
        queueCondition.notify_one();
    }

    return true;
}

int FrameRecorder::getFrameCount() const noexcept {
    return currentSession ? currentSession->frameCounter.load() : 0;
}

void FrameRecorder::setRecordingOptions(const RecordingOptions& newOptions) {
    if (isRecordingActive) {
        spdlog::warn("Cannot change options while recording is active");
        return;
    }

    options = newOptions;
}

const FrameRecorder::RecordingOptions& FrameRecorder::getRecordingOptions() const noexcept {
    return options;
}

// Helper function to clean up completed operations
void FrameRecorder::cleanupCompletedOperations() {
    std::lock_guard<std::mutex> lock(pendingOpsMutex);

    for (auto it = pendingOperations.begin(); it != pendingOperations.end();) {
        if (it->second.valid() &&
            it->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            it = pendingOperations.erase(it);
        } else {
            ++it;
        }
    }
}

// Wait for operations to complete if we've reached the limit
void FrameRecorder::waitForOperationsIfNeeded() {
    std::lock_guard<std::mutex> lock(pendingOpsMutex);

    // First try to cleanup completed operations
    for (auto it = pendingOperations.begin(); it != pendingOperations.end();) {
        if (it->second.valid() &&
            it->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            it = pendingOperations.erase(it);
        } else {
            ++it;
        }
    }

    // If we still have too many operations, wait for the oldest one
    if (pendingOperations.size() >= maxConcurrentOperations && !pendingOperations.empty()) {
        // Find the oldest operation (lowest ID)
        auto oldestOp = std::min_element(pendingOperations.begin(), pendingOperations.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        if (oldestOp != pendingOperations.end() && oldestOp->second.valid()) {
            // Wait for it to complete
            oldestOp->second.wait();
            pendingOperations.erase(oldestOp);
        }
    }
}

// Process frames from the queue
void FrameRecorder::processFrameQueue() {
    spdlog::debug("Frame processing thread started");

    int processedFrames = 0;
    uint64_t operationId = 0; // For tracking operations

    while (shouldProcessFrames) {
        FrameData frame;

        // Get a frame from the queue
        {
            std::unique_lock<std::mutex> lock(queueMutex);

            if (!frameQueue.empty()) {
                frame = std::move(frameQueue.front());
                frameQueue.pop();
                queueCondition.notify_all(); // Notify if queue was full
            } else if (!isRecordingActive) {
                // No more frames and not recording
                break;
            } else {
                // Wait for more frames
                queueCondition.wait(lock);
                continue;
            }
        }

        if (currentSession) {
            // Get frame index and increment counter
            int frameIndex = currentSession->frameCounter.fetch_add(1);
            processedFrames++;

            // Process the frame
            try {
                // Wait if we have too many pending operations
                waitForOperationsIfNeeded();

                // Save frame to disk using the thread pool
                saveFrameToDisk(frame, frameIndex, operationId++);

                // Log progress periodically
                if (processedFrames % 30 == 0) {
                    auto elapsedSec = std::chrono::duration<double>(
                        std::chrono::system_clock::now() - currentSession->startTime).count();
                    double fps = processedFrames / elapsedSec;

                    // Also report pending operations
                    std::lock_guard<std::mutex> lock(pendingOpsMutex);
                    spdlog::info("Processed {} frames ({:.1f} FPS), Pending I/O: {}, Queue: {}",
                               processedFrames, fps, pendingOperations.size(), frameQueue.size());
                }

                // Periodically clean up completed operations
                if (processedFrames % 10 == 0) {
                    cleanupCompletedOperations();
                }
            } catch (const std::exception& e) {
                spdlog::error("Error processing frame {}: {}", frameIndex, e.what());
            }
        }
    }

    spdlog::debug("Frame processing thread finished after processing {} frames", processedFrames);
}

std::string FrameRecorder::generateUniqueSessionId(const std::string& sessionId) const {
    const auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::system_clock::to_time_t(now);
    tm t{};

#ifdef _WIN32
    localtime_s(&t, &timestamp);
#else
    localtime_r(&timestamp, &t);
#endif

    std::stringstream ss;

    if (sessionId.empty())
    {
        ss << "recording_" << std::put_time(&t, "%Y%m%d_%H%M%S");
    } else
    {
        ss << sessionId << std::put_time(&t, "%Y%m%d_%H%M%S");
    }

    return ss.str();
}

cv::Mat FrameRecorder::getFrameFromSource(int frameIndex) const
{
    cv::Mat frame;

    // Try processing_temp first
    if (fs::exists(currentSession->outputPath / "processing_temp")) {
        std::string framePath = (currentSession->outputPath / "processing_temp" /
            ("frame_" + std::to_string(frameIndex) + ".png")).string();
        if (fs::exists(framePath)) {
            frame = cv::imread(framePath);
            if (!frame.empty()) {
                return frame;
            }
        }
    } else {
        spdlog::error("Directory processing_temp does not have the necessary files");
    }

    return frame;
}

bool FrameRecorder::stopRecording() {
    if (!isRecordingActive || !currentSession) {
        return false;
    }

    try {
        spdlog::info("Stopping recording...");

        // First mark as not recording to stop accepting new frames
        isRecordingActive = false;

        // Wait for queue to drain with timeout
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            auto waitResult = queueCondition.wait_for(lock,
                                std::chrono::seconds(5), // 5 second timeout
                                [this] { return frameQueue.empty(); });

            if (!waitResult) {
                // Timed out - log the warning and continue
                spdlog::warn("Timeout waiting for frame queue to empty, proceeding with {} frames in queue",
                           frameQueue.size());
                // Clear queue to prevent further processing
                std::queue<FrameData>().swap(frameQueue); // Efficiently clear the queue
            }
        }

        // Safely stop processing thread
        shouldProcessFrames = false;
        queueCondition.notify_all();

        if (processingThread.joinable()) {
            // Create a future to handle thread joining with timeout
            std::future<void> joinFuture = std::async(std::launch::async, [&]() {
                if (processingThread.joinable()) {
                    processingThread.join();
                }
            });

            // Wait with timeout
            if (joinFuture.wait_for(std::chrono::seconds(3)) == std::future_status::timeout) {
                spdlog::warn("Processing thread join timed out, proceeding with shutdown");
                // We can't safely terminate the thread, so we'll leak it
                // This is safer than crashing
            }
        }

        // Handle pending operations with extreme caution
        try {
            std::lock_guard<std::mutex> lock(pendingOpsMutex);
            spdlog::info("Handling {} pending file operations...", pendingOperations.size());

            // Don't try to wait for all operations - it could hang
            // Just log and clear them
            pendingOperations.clear();
            spdlog::info("Cleared pending operations");
        } catch (const std::exception& e) {
            spdlog::error("Exception clearing pending operations: {}", e.what());
            // Continue shutdown anyway
        }

        // Finalize recording data
        auto endTime = std::chrono::system_clock::now();
        auto end_time_t = std::chrono::system_clock::to_time_t(endTime);
        std::chrono::duration<double> duration = endTime - currentSession->startTime;
        int totalFrames = currentSession->frameCounter;
        double actualFps = totalFrames > 0 ? totalFrames / duration.count() : 0.0;

        // Close metadata file safely
        try {
            if (currentSession->metadataStream.is_open()) {
                // Write basic summary data
                tm timeinfo;
#ifdef _WIN32
                localtime_s(&timeinfo, &end_time_t);
#else
                localtime_r(&end_time_t, &timeinfo);
#endif

                currentSession->metadataStream << "Recording ended at: "
                    << std::put_time(&timeinfo, "%Y-%m-%d %H:%M:%S") << std::endl
                    << "Total frames: " << totalFrames << std::endl
                    << "Recording duration (seconds): " << duration.count() << std::endl
                    << "Actual FPS: " << actualFps << std::endl;
                currentSession->metadataStream.close();
            }
        } catch (const std::exception& e) {
            spdlog::error("Error writing metadata: {}", e.what());
            // Continue shutdown
        }

        // Save frame timestamps for analysis
        try {
            if (!currentSession->frameTimestamps.empty()) {
                std::string timestampPath = (currentSession->outputPath / "frame_timestamps.csv").string();
                std::ofstream timestampFile(timestampPath);

                if (timestampFile.is_open()) {
                    // Write CSV header
                    timestampFile << "frame_index,timestamp_ns,elapsed_ms,delta_ms\n";

                    int64_t startTs = currentSession->frameTimestamps.front();
                    int64_t prevTs = startTs;

                    for (size_t i = 0; i < currentSession->frameTimestamps.size(); i++) {
                        int64_t ts = currentSession->frameTimestamps[i];
                        int64_t elapsedMs = (ts - startTs) / 1000000;
                        int64_t deltaMs = (ts - prevTs) / 1000000;

                        timestampFile << i << "," << ts << "," << elapsedMs << "," << deltaMs << std::endl;
                        prevTs = ts;
                    }

                    timestampFile.close();
                    spdlog::info("Timestamp data saved to: {}", timestampPath);
                } else {
                    spdlog::error("Failed to open timestamp file for writing: {}", timestampPath);
                }
            }
        } catch (const std::exception& e) {
            spdlog::error("Error saving timestamp data: {}", e.what());
            // Continue shutdown
        }

        // Generate video completely separated from I/O operations
        try {
            if (totalFrames > 0 && options.generateVideoOnRecording) {
                createRecordingVideo(totalFrames, actualFps, duration);
            }
        } catch (const std::exception& e) {
            spdlog::error("Error creating video: {}", e.what());
            // Continue shutdown
        }

        std::string timingPath = (currentSession->outputPath / "video_timing.txt").string();
        std::ofstream timingFile(timingPath);
        if (timingFile.is_open())
        {
            timingFile << "Original recording information:" << std::endl;
            timingFile << "  Total frames: " << totalFrames << std::endl;
            timingFile << "  Duration: " << duration.count() << " seconds" << std::endl;
            timingFile << "  Actual FPS: " << actualFps << std::endl << std::endl;
        }

        // Log success even if some parts failed
        spdlog::info("Recording stopped. Captured {} frames over {:.2f} seconds ({:.2f} FPS)",
                    totalFrames, duration.count(), actualFps);

        return true;
    } catch (const std::exception& e) {
        spdlog::error("Critical error during stopRecording: {}", e.what());
        return false;
    } catch (...) {
        spdlog::error("Unknown error during stopRecording");
        return false;
    }
}

void FrameRecorder::createRecordingVideo(int totalFrames, double actualFps, std::chrono::duration<double> duration ) {
    spdlog::info("Creating video file...");
    if (totalFrames > 0) {
        // Determine the video path and parameters based on compression option
        std::string videoPath;
        double targetFps;
        bool duplicateFrames;
        std::string videoType;

        if (options.useVideoCompression) {
            // Compressed video uses actual FPS, no frame duplication
            videoPath = (currentSession->outputPath / "videoCompressed.mp4").string();
            targetFps = actualFps;  // Use actual FPS for compressed video
            duplicateFrames = false;
            videoType = "compressed";
        } else {
            // Standard video uses 30 FPS with frame duplication
            videoPath = (currentSession->outputPath / "videoStandard.mp4").string();
            targetFps = 30.0;  // Standard FPS
            duplicateFrames = true;
            videoType = "standard";
        }

        // Common code to find the first frame
        cv::Mat firstFrame;
        bool useProcessingTemp = fs::exists(currentSession->outputPath / "processing_temp");

        // Try to load the first frame
        if (useProcessingTemp) {
            std::string firstFramePath = (currentSession->outputPath / "processing_temp" / "frame_0.png").string();
            if (fs::exists(firstFramePath)) {
                firstFrame = cv::imread(firstFramePath);
                spdlog::info("Using frames from processing_temp for video creation");
            }
        }

        // Search for any available frame
        if (firstFrame.empty()) {
            for (int i = 1; i < totalFrames; i++) {
                if (useProcessingTemp) {
                    std::string framePath = (currentSession->outputPath / "processing_temp" /
                        ("frame_" + std::to_string(i) + ".png")).string();
                    if (fs::exists(framePath)) {
                        firstFrame = cv::imread(framePath);
                        break;
                    }
                }
            }
        }

        if (firstFrame.empty()) {
            spdlog::error("Could not find any frames to create video!");
        } else {
            // Common code for video writer setup
            std::vector<std::pair<std::string, int>> codecs = {
                {"XVID", cv::VideoWriter::fourcc('X', 'V', 'I', 'D')},
                {"MJPG", cv::VideoWriter::fourcc('M', 'J', 'P', 'G')},
                {"MP4V", cv::VideoWriter::fourcc('M', 'P', '4', 'V')}
            };

            cv::VideoWriter videoWriter;
            bool videoWriterCreated = false;

            // Try different codecs
            for (const auto& codec : codecs) {
                spdlog::info("Trying codec: {}", codec.first);
                videoWriter.open(videoPath, codec.second, targetFps, firstFrame.size());

                if (videoWriter.isOpened()) {
                    videoWriterCreated = true;
                    spdlog::info("Successfully opened video writer with codec: {}", codec.first);
                    break;
                }
            }

            if (!videoWriterCreated) {
                spdlog::error("Failed to create video writer at {} with any codec", videoPath);
            } else {
                spdlog::info("Creating {} MP4 video with FPS: {:.2f}", videoType, targetFps);

                // Setup for frame processing
                std::unordered_map<int, cv::Mat> frameCache;
                cv::Mat currentFrame;
                fs::path frameSourceDir = currentSession->outputPath / "processing_temp";
                int framesWritten = 0;

                // Different logic based on whether we're duplicating frames
                if (duplicateFrames) {
                    // Calculate frame duplication factor
                    double frameRepeatFactor = targetFps / actualFps;
                    int totalOutputFrames = static_cast<int>(std::ceil(totalFrames * frameRepeatFactor));

                    spdlog::info("Frame repeat factor: {:.2f}, output will have approximately {} frames",
                               frameRepeatFactor, totalOutputFrames);

                    // Create timing map for duplicated frames
                    std::vector<int> frameMap(totalOutputFrames);
                    for (int outFrame = 0; outFrame < totalOutputFrames; outFrame++) {
                        double sourceFrameExact = outFrame / frameRepeatFactor;
                        int sourceFrame = std::min(static_cast<int>(sourceFrameExact), totalFrames - 1);
                        frameMap[outFrame] = sourceFrame;
                    }

                    // Track the last frame to avoid reloading
                    int lastFrame = -1;

                    // Process frames with duplication
                    for (int outFrame = 0; outFrame < totalOutputFrames; outFrame++) {
                        int sourceFrame = frameMap[outFrame];

                        // Only load a new frame from disk if needed
                        if (sourceFrame != lastFrame) {
                            // Load frame (using cache if available)
                            auto cacheIt = frameCache.find(sourceFrame);
                            if (cacheIt != frameCache.end()) {
                                currentFrame = cacheIt->second;
                            } else {
                                std::string framePath = (frameSourceDir /
                                    ("frame_" + std::to_string(sourceFrame) + ".png")).string();
                                if (fs::exists(framePath)) {
                                    currentFrame = cv::imread(framePath);
                                    // Cache frame
                                    if (frameCache.size() < 10) {
                                        frameCache[sourceFrame] = currentFrame;
                                    }
                                } else {
                                    // If frame is missing, use the last valid frame
                                    spdlog::warn("Frame {} not found, using previous frame", sourceFrame);
                                    if (currentFrame.empty() && !frameCache.empty()) {
                                        currentFrame = frameCache.begin()->second;
                                    }
                                }
                            }
                            lastFrame = sourceFrame;
                        }

                        // Write the frame
                        if (!currentFrame.empty()) {
                            videoWriter.write(currentFrame);
                            framesWritten++;

                            // Log progress periodically
                            if (outFrame % 100 == 0 || outFrame == totalOutputFrames - 1) {
                                spdlog::info("Video creation progress: {:.1f}% ({}/{})",
                                          (100.0 * outFrame) / totalOutputFrames,
                                          outFrame + 1, totalOutputFrames);
                            }
                        }
                    }
                } else {
                    // Process frames without duplication (compressed version)
                    for (int frameIndex = 0; frameIndex < totalFrames; frameIndex++) {
                        // Load frame (using cache if available)
                        auto cacheIt = frameCache.find(frameIndex);
                        if (cacheIt != frameCache.end()) {
                            currentFrame = cacheIt->second;
                        } else {
                            std::string framePath = (frameSourceDir /
                                ("frame_" + std::to_string(frameIndex) + ".png")).string();
                            if (fs::exists(framePath)) {
                                currentFrame = cv::imread(framePath);
                                // Cache frame
                                if (frameCache.size() < 10) {
                                    frameCache[frameIndex] = currentFrame;
                                }
                            } else {
                                // Skip if frame is missing
                                spdlog::warn("Frame {} not found, skipping", frameIndex);
                                continue;
                            }
                        }

                        // Write the frame
                        if (!currentFrame.empty()) {
                            videoWriter.write(currentFrame);
                            framesWritten++;

                            // Log progress periodically
                            if (frameIndex % 100 == 0 || frameIndex == totalFrames - 1) {
                                spdlog::info("Video creation progress: {:.1f}% ({}/{})",
                                          (100.0 * frameIndex) / totalFrames,
                                          frameIndex + 1, totalFrames);
                            }
                        }
                    }
                }

                videoWriter.release();
                spdlog::info("Video created successfully with {} frames", framesWritten);

                // Save timing information for reference
                std::string timingPath = (currentSession->outputPath / "video_timing.txt").string();
                std::ofstream timingFile(timingPath);
                if (timingFile.is_open()) {

                    timingFile << "Video information:" << std::endl;
                    if (duplicateFrames) {
                        timingFile << "  Type: Standard video with frame duplication" << std::endl;
                        timingFile << "  Standard FPS: " << targetFps << std::endl;
                        timingFile << "  Frame repeat factor: " << (targetFps / actualFps) << std::endl;
                    } else {
                        timingFile << "  Type: Compressed video with original timing" << std::endl;
                        timingFile << "  Used FPS: " << targetFps << std::endl;
                        timingFile << "  No frame duplication - using original timing" << std::endl;
                    }
                    timingFile << "  Total frames: " << framesWritten << std::endl;
                    timingFile << "  Output duration: " << (framesWritten / targetFps) << " seconds" << std::endl;
                    timingFile.close();
                }
            }
        }
    }
}

bool FrameRecorder::startRecording(const std::string& sessionId) {
    if (isRecordingActive) {
        spdlog::warn("Recording already in progress");
        return false;
    }

    // Create a new session
    currentSession = std::make_unique<RecordingSession>();
    currentSession->startTime = std::chrono::system_clock::now();
    currentSession->lastFrameTime = currentSession->startTime;
    currentSession->frameCounter = 0;

    // Create a unique session ID if not provided
    std::string actualSessionId = generateUniqueSessionId(sessionId);

    // Create output directory
    std::string outputDir = options.outputDirectory + "/" + actualSessionId;
    currentSession->outputPath = fs::absolute(outputDir);

    try {
        // Create necessary directories
        fs::create_directories(currentSession->outputPath);
        fs::create_directories(currentSession->outputPath / "depth_raw");

        // Create processing_temp directory if saving original frames
        if (options.saveOriginalFrames) {
            fs::create_directories(currentSession->outputPath / "processing_temp");
        }

        // Create metadata file
        std::string metadataPath = (currentSession->outputPath / "metadata.txt").string();
        currentSession->metadataStream.open(metadataPath);

        if (!currentSession->metadataStream.is_open()) {
            spdlog::error("Failed to create metadata file at: {}", metadataPath);
            return false;
        }

        // Write metadata header
        const auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::system_clock::to_time_t(now);
        struct tm t;

#ifdef _WIN32
        localtime_s(&t, &timestamp);
#else
        localtime_r(&timestamp, &t);
#endif

        currentSession->metadataStream << "Recording started at: "
            << std::put_time(&t, "%Y-%m-%d %H:%M:%S") << std::endl;
        currentSession->metadataStream << "Session ID: " << actualSessionId << std::endl;
        currentSession->metadataStream << "Output directory: " << currentSession->outputPath.string() << std::endl;
        currentSession->metadataStream << "Target FPS: " << options.targetFps << std::endl;
        currentSession->metadataStream << "Frame rate limiting: " << (options.limitFrameRate ? "enabled" : "disabled") << std::endl;
        currentSession->metadataStream << "Video compression: " << (options.useVideoCompression ? "enabled" : "disabled") << std::endl;
        currentSession->metadataStream << "Process every N frames: " << options.processEveryNFrames << std::endl;
        currentSession->metadataStream << "Save original frames: " << (options.saveOriginalFrames ? "enabled" : "disabled") << std::endl;
        currentSession->metadataStream << "Using optimized I/O with thread pool: Yes" << std::endl;
        currentSession->metadataStream << "I/O threads: " << (ioPool ? "4" : "0") << std::endl;
        currentSession->metadataStream << "Max concurrent operations: " << maxConcurrentOperations << std::endl;

        // Start the processing thread
        shouldProcessFrames = true;

        if (processingThread.joinable()) {
            processingThread.join();
        }

        processingThread = std::thread(&FrameRecorder::processFrameQueue, this);

        // Clear any pending operations from previous recordings
        {
            std::lock_guard<std::mutex> lock(pendingOpsMutex);
            pendingOperations.clear();
        }

        // Mark as recording
        isRecordingActive = true;

        spdlog::info("Recording started to {} with optimized I/O", currentSession->outputPath.string());
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to start recording: {}", e.what());
        currentSession.reset();
        return false;
    }
}

// Helper functions for async file operations
static void saveColorToDisk(const cv::Mat& colorImage, const std::string& path, int quality, bool useJpeg) {
    std::vector<int> params;
    std::string actualPath = path;

    if (useJpeg) {
        actualPath = path.substr(0, path.find_last_of('.')) + ".jpg";
        params = {cv::IMWRITE_JPEG_QUALITY, quality };
    } else {
        params = {cv::IMWRITE_PNG_COMPRESSION, 1};
    }

    imwrite(actualPath, colorImage, params);
}

static void saveDepthToDisk(const cv::Mat& depthImage, const std::string& path) {
    int rows = depthImage.rows;
    int cols = depthImage.cols;

    std::ofstream depthFile(path, std::ios::binary);
    if (depthFile.is_open()) {
        // Write dimensions
        depthFile.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        depthFile.write(reinterpret_cast<const char*>(&cols), sizeof(int));

        // Write raw depth data
        depthFile.write(reinterpret_cast<const char*>(depthImage.data),
                      depthImage.total() * depthImage.elemSize());

        depthFile.close();
    }
}



// Save frame to disk using the thread pool
bool FrameRecorder::saveFrameToDisk(const FrameData& frame, int frameIndex, uint64_t operationId) {
    if (!currentSession) {
        return false;
    }

    bool success = true;

    // Always save original frames if enabled (primary source of truth)
    if (options.saveOriginalFrames) {
        try {
            std::string origPath = (currentSession->outputPath / "processing_temp" /
                ("frame_" + std::to_string(frameIndex) + ".png")).string();

            // Queue the color image save operation
            std::future<void> colorFuture;
            if (ioPool) {
                colorFuture = ioPool->enqueue([frame, origPath, this]() {
                    saveColorToDisk(frame.colorImage, origPath, options.jpegQuality, options.useJpegForColorFrames);
                });

                // Track this operation
                {
                    std::lock_guard<std::mutex> lock(pendingOpsMutex);
                    pendingOperations[operationId] = std::move(colorFuture);
                }
            } else {
                // Fallback to synchronous save
                saveColorToDisk(frame.colorImage, origPath, options.jpegQuality, options.useJpegForColorFrames);
            }
        } catch (const std::exception& e) {
            spdlog::error("Failed to save color image: {}", e.what());
            success = false;
        }
    }

    // Save depth data for processing
    if (frame.depthImage.empty() || frame.depthImage.type() != CV_16UC1) {
        spdlog::warn("Invalid depth image for frame {}", frameIndex);
        success = false;
    } else {
        // Save raw depth data
        try {
            std::string depthPath = (currentSession->outputPath / "depth_raw" /
                ("frame_" + std::to_string(frameIndex) + ".bin")).string();

            // Queue the depth data save operation
            std::future<void> depthFuture;
            if (ioPool) {
                depthFuture = ioPool->enqueue([frame, depthPath]() {
                    saveDepthToDisk(frame.depthImage, depthPath);
                });

                // Track this operation
                {
                    std::lock_guard<std::mutex> lock(pendingOpsMutex);
                    pendingOperations[operationId + 1000000] = std::move(depthFuture); // Offset to avoid ID collision
                }
            } else {
                // Fallback to synchronous save
                saveDepthToDisk(frame.depthImage, depthPath);
            }

            // Only index frames that should be processed
        } catch (const std::exception& e) {
            spdlog::error("Failed to save depth data: {}", e.what());
            success = false;
        }
    }

    // Update metadata (keep this synchronous for consistency)
    if (currentSession->metadataStream.is_open()) {
        static std::mutex metadataMutex;
        std::lock_guard<std::mutex> lock(metadataMutex);

        auto timestamp = std::chrono::system_clock::to_time_t(frame.timestamp);
        tm buf;
#ifdef _WIN32
        gmtime_s(&buf, &timestamp);
#else
        gmtime_r(&timestamp, &buf);
#endif
        currentSession->metadataStream << "Frame: " << frameIndex << " Timestamp: "
            << std::put_time(&buf, "%H:%M:%S.")
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   frame.timestamp.time_since_epoch()).count() % 1000
            << std::endl;
    }

    return success;
}

void FrameRecorder::cleanupProcessingTemp(const std::string& sessionId) {
    // Get the session directory
    std::string sessionDir = options.outputDirectory + "/" + sessionId;
    fs::path processingTempDir = fs::path(sessionDir) / "processing_temp";

    try {
        if (fs::exists(processingTempDir)) {
            spdlog::info("Cleaning up processing_temp directory: {}", processingTempDir.string());
            fs::remove_all(processingTempDir);
            spdlog::info("Cleanup complete");
        } else {
            spdlog::info("No processing_temp directory found at: {}", processingTempDir.string());
        }
    } catch (const fs::filesystem_error& e) {
        spdlog::error("Error cleaning up processing_temp: {}", e.what());
    }
}