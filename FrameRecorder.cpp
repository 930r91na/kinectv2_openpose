#include "FrameRecorder.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <algorithm>

namespace fs = std::filesystem;

FrameRecorder::FrameRecorder(RecordingOptions opts)
    : options(std::move(opts)) {

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
        stopRecording();
    }

    // Make sure processing thread is stopped
    shouldProcessFrames = false;
    queueCondition.notify_all();

    if (processingThread.joinable()) {
        processingThread.join();
    }
}

FrameRecorder::FrameRecorder(FrameRecorder&& other) noexcept
    : options(std::move(other.options)),
      currentSession(std::move(other.currentSession)),
      isRecordingActive(other.isRecordingActive.load()),
      shouldProcessFrames(other.shouldProcessFrames.load()) {

    // Move queued frames
    {
        std::lock_guard<std::mutex> lock(other.queueMutex);
        frameQueue = std::move(other.frameQueue);
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

        // Move data from other
        options = std::move(other.options);
        currentSession = std::move(other.currentSession);
        isRecordingActive = other.isRecordingActive.load();
        shouldProcessFrames = other.shouldProcessFrames.load();

        // Move queued frames
        {
            std::lock_guard<std::mutex> lock(other.queueMutex);
            frameQueue = std::move(other.frameQueue);
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

    // Create a new frame data object
    FrameData frameData;
    frameData.colorImage = std::move(const_cast<cv::Mat&>(colorFrame));
    frameData.depthImage = std::move(const_cast<cv::Mat&>(depthFrame));
    frameData.timestamp = std::chrono::system_clock::now();

    // Store timestamp
    auto timestampNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        frameData.timestamp.time_since_epoch()).count();
    currentSession->frameTimestamps.push_back(timestampNs);

    // Add to processing queue
    {
        std::unique_lock<std::mutex> lock(queueMutex);

        // Wait if queue is full to prevent excessive memory usage
        queueCondition.wait(lock, [this] {
            return frameQueue.size() < MAX_QUEUE_SIZE || !isRecordingActive;
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

std::vector<FrameRecorder::FrameData> FrameRecorder::loadRecordedFrames(const std::string& directory) {
    std::vector<FrameData> frames;

    // Validate directory
    if (!fs::exists(directory)) {
        spdlog::error("Directory does not exist: {}", directory);
        return frames;
    }

    // Check for required directories
    bool hasVideoFile = fs::exists(directory + "/videoStandard.mp4") || fs::exists(directory + "/videoCompressed.mp4");
    bool hasProcessingDir = fs::exists(directory + "/processing_temp");

    // Check for processing temp directory first as it contains uncompressed originals
    fs::path frameSourceDir;
    bool useProcessingTemp = false;

    if (hasProcessingDir) {
        frameSourceDir = directory + "/processing_temp";
        useProcessingTemp = true;
        spdlog::info("Using processing_temp directory for frames");
    } else if (hasVideoFile) {
        frameSourceDir = directory;
        spdlog::info("Using video file for frames");
    } else {
        spdlog::error("No valid frame source found in {}", directory);
        return frames;
    }

    // Get list of frames to load
    std::vector<int> framesToLoad;

    // First check if we have a processing_frames.txt file
    if (fs::exists(directory + "/process_frames.txt")) {
        try {
            std::ifstream frameListFile(directory + "/process_frames.txt");
            if (frameListFile.is_open()) {
                std::string line;
                while (std::getline(frameListFile, line)) {
                    try {
                        int frameIndex = std::stoi(line);
                        framesToLoad.push_back(frameIndex);
                    } catch (...) {
                        // Skip invalid lines
                    }
                }
                frameListFile.close();
                spdlog::info("Loaded {} frames from process_frames.txt", framesToLoad.size());
            }
        } catch (const std::exception& e) {
            spdlog::error("Error reading process_frames.txt: {}", e.what());
        }
    }

    // If no specific list, scan directory
    if (framesToLoad.empty()) {
        fs::path scanDir = useProcessingTemp ? frameSourceDir : directory + "/depth_raw";

        // Scan for file pattern
        for (const auto& entry : fs::directory_iterator(scanDir)) {
            std::string filename = entry.path().filename().string();
            // Extract frame number from "frame_X.bin" or "frame_X.png"
            size_t underscorePos = filename.find('_');
            size_t dotPos = filename.find('.');

            if (underscorePos != std::string::npos && dotPos != std::string::npos) {
                std::string indexStr = filename.substr(underscorePos + 1, dotPos - underscorePos - 1);
                try {
                    int frameIndex = std::stoi(indexStr);
                    framesToLoad.push_back(frameIndex);
                } catch (...) {
                    // Skip files with invalid indices
                }
            }
        }
    }

    std::ranges::sort(framesToLoad);
    spdlog::info("Found {} depth frame files", framesToLoad.size());

    if (framesToLoad.empty()) {
        spdlog::error("No frames found to load");
        return frames;
    }

    // Prepare to load frames
    frames.reserve(framesToLoad.size());

    // Open video capture if using video
    cv::VideoCapture videoCapture;
    if (hasVideoFile && !useProcessingTemp) {
        videoCapture.open(directory + "/videoStandard.mp4");
        if (!videoCapture.isOpened()) {
            videoCapture.open(directory + "/videoCompressed.mp4");
            if (!videoCapture.isOpened())
            {
                spdlog::error("Failed to open color video file");
                return frames;
            }
        }
    }

    // Load each frame
    for (int frameIdx : framesToLoad) {
        FrameData frame;

        // Load color image
        if (useProcessingTemp) {
            std::string colorPath = (frameSourceDir / ("frame_" + std::to_string(frameIdx) + ".png")).string();
            frame.colorImage = cv::imread(colorPath);
            if (frame.colorImage.empty()) {
                spdlog::warn("Failed to load color frame {} from processing_temp", frameIdx);
                continue;
            }
        } else if (hasVideoFile) {
            videoCapture.set(cv::CAP_PROP_POS_FRAMES, frameIdx);
            if (!videoCapture.read(frame.colorImage)) {
                spdlog::warn("Failed to read color frame {} from video", frameIdx);
                continue;
            }
        }

        // Load depth data
        std::string depthPath = (fs::path(directory) / "depth_raw" / ("frame_" + std::to_string(frameIdx) + ".bin")).string();
        std::ifstream depthFile(depthPath, std::ios::binary);

        if (depthFile.is_open()) {
            // Read dimensions
            int rows, cols;
            depthFile.read(reinterpret_cast<char*>(&rows), sizeof(int));
            depthFile.read(reinterpret_cast<char*>(&cols), sizeof(int));

            // Read data
            frame.depthImage = cv::Mat(rows, cols, CV_16UC1);
            depthFile.read(reinterpret_cast<char*>(frame.depthImage.data),
                         frame.depthImage.total() * frame.depthImage.elemSize());

            depthFile.close();
        } else {
            spdlog::warn("Failed to load depth data for frame {}", frameIdx);
            continue;
        }

        // Set timestamp - just use frame index as we don't have actual timestamps
        frame.timestamp = std::chrono::system_clock::now() +
            std::chrono::milliseconds(frameIdx * 33); // Approximate 30fps

        frames.push_back(std::move(frame));

        // Log progress for large datasets
        if (frames.size() % 100 == 0) {
            spdlog::info("Loaded {}/{} frames", frames.size(), framesToLoad.size());
        }
    }

    if (hasVideoFile && !useProcessingTemp) {
        videoCapture.release();
    }

    spdlog::info("Successfully loaded {} frames", frames.size());
    return frames;
}

void FrameRecorder::processFrameQueue() {
    spdlog::debug("Frame processing thread started");

    int processedFrames = 0;

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
                // Save frame to disk
                saveFrameToDisk(frame, frameIndex);

                // Log progress periodically
                if (processedFrames % 30 == 0) {
                    auto elapsedSec = std::chrono::duration<double>(
                        std::chrono::system_clock::now() - currentSession->startTime).count();
                    double fps = processedFrames / elapsedSec;

                    spdlog::info("Processed {} frames ({:.1f} FPS)", processedFrames, fps);
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
    }else
    {
        spdlog::error("Directory processing_temp does not have the necessary files");
    }

    return frame;
}

bool FrameRecorder::stopRecording() {
    if (!isRecordingActive || !currentSession) {
        return false;
    }

    spdlog::info("Stopping recording...");

    // Mark as not recording
    isRecordingActive = false;

    // Wait for all frames to be processed
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCondition.wait(lock, [this] { return frameQueue.empty(); });
    }

    // Finalize recording
    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = endTime - currentSession->startTime;

    // Calculate actual FPS
    int totalFrames = currentSession->frameCounter;
    double actualFps = totalFrames / duration.count();

    // Write final metadata
    if (currentSession->metadataStream.is_open()) {
        auto end_time_t = std::chrono::system_clock::to_time_t(endTime);
        tm t;
#ifdef _WIN32
        localtime_s(&t, &end_time_t);
#else
        localtime_r(&end_time_t, &t);
#endif
        currentSession->metadataStream << "Recording ended at: "
            << std::put_time(&t, "%Y-%m-%d %H:%M:%S") << std::endl;
        currentSession->metadataStream << "Total frames: " << totalFrames << std::endl;
        currentSession->metadataStream << "Recording duration (seconds): " << duration.count() << std::endl;
        currentSession->metadataStream << "Actual FPS: " << actualFps << std::endl;
        currentSession->metadataStream.close();
    }

    // Video creation code - for both compressed and standard formats
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
                    timingFile << "Original recording information:" << std::endl;
                    timingFile << "  Total frames: " << totalFrames << std::endl;
                    timingFile << "  Duration: " << duration.count() << " seconds" << std::endl;
                    timingFile << "  Actual FPS: " << actualFps << std::endl << std::endl;

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

    // Save frame timestamps for analysis
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
        }
    }

    // Save process_frames.txt with the indices of original frames
    // (not the duplicates created for video)
    if (options.saveOriginalFrames) {
        std::string processFramesPath = (currentSession->outputPath / "process_frames.txt").string();
        std::ofstream processFramesFile(processFramesPath);

        if (processFramesFile.is_open()) {
            // Look at frameTimestamps to identify original frames
            if (!currentSession->frameTimestamps.empty()) {
                constexpr int64_t MIN_DELTA_NS = 50 * 1000000; // 50ms in nanoseconds
                int64_t lastTs = currentSession->frameTimestamps[0];

                // First frame is always original
                processFramesFile << "0" << std::endl;

                for (size_t i = 1; i < currentSession->frameTimestamps.size(); i++) {
                    int64_t ts = currentSession->frameTimestamps[i];
                    int64_t delta = ts - lastTs;

                    // If significant time has passed, this is an original frame
                    if (delta >= MIN_DELTA_NS) {
                        processFramesFile << i << std::endl;
                        lastTs = ts;
                    }
                }
            } else {
                // If no timestamps, just use every frame
                for (int i = 0; i < totalFrames; i++) {
                    processFramesFile << i << std::endl;
                }
            }

            processFramesFile.close();
            spdlog::info("Created process_frames.txt with original frame indices");
        }
    }

    // Generate a summary file
    std::string summaryPath = (currentSession->outputPath / "recording_summary.txt").string();
    std::ofstream summaryFile(summaryPath);

    if (summaryFile.is_open()) {
        summaryFile << "Recording Summary" << std::endl;
        summaryFile << "----------------" << std::endl;
        summaryFile << "Session ID: " << currentSession->outputPath.filename().string() << std::endl;
        summaryFile << "Total frames: " << totalFrames << std::endl;
        summaryFile << "Recording duration: " << duration.count() << " seconds" << std::endl;
        summaryFile << "Actual FPS: " << actualFps << std::endl;
        summaryFile << "Target FPS: " << options.targetFps << std::endl;
        summaryFile << "Output directory: " << currentSession->outputPath.string() << std::endl;
        summaryFile << "Original frames saved: " << (options.saveOriginalFrames ? "Yes" : "No") << std::endl;

        // Count files to verify
        try {
            if (options.useVideoCompression &&
                fs::exists(currentSession->outputPath / "colorCompressed.mp4")) {
                summaryFile << "Color format: MP4 video" << std::endl;
                summaryFile << "Video playback: Standard 30 FPS with timing preserved" << std::endl;
            }

            if (fs::exists(currentSession->outputPath / "depth_raw")) {
                size_t depthFrames = 0;
                for (const auto& entry : fs::directory_iterator(currentSession->outputPath / "depth_raw")) {
                    if (entry.path().extension() == ".bin") {
                        depthFrames++;
                    }
                }
                summaryFile << "Depth frames: " << depthFrames << std::endl;
            }

            if (fs::exists(currentSession->outputPath / "processing_temp")) {
                size_t tempFrames = 0;
                for (const auto& entry : fs::directory_iterator(currentSession->outputPath / "processing_temp")) {
                    if (entry.path().extension() == ".png") {
                        tempFrames++;
                    }
                }
                summaryFile << "Processing temp frames: " << tempFrames << std::endl;
            }
        } catch (const fs::filesystem_error& e) {
            summaryFile << "Error counting files: " << e.what() << std::endl;
        }

        summaryFile.close();
    }

    spdlog::info("Recording stopped. Captured {} frames over {:.2f} seconds ({:.2f} FPS)",
                totalFrames, duration.count(), actualFps);

    // Return recording successful
    return true;
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

        // Start the processing thread
        shouldProcessFrames = true;

        if (processingThread.joinable()) {
            processingThread.join();
        }

        processingThread = std::thread(&FrameRecorder::processFrameQueue, this);

        // Mark as recording
        isRecordingActive = true;

        spdlog::info("Recording started to {}", currentSession->outputPath.string());
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to start recording: {}", e.what());
        currentSession.reset();
        return false;
    }
}

bool FrameRecorder::saveFrameToDisk(const FrameData& frame, int frameIndex) {
    if (!currentSession) {
        return false;
    }

    bool success = true;

    // Always save original frames if enabled (primary source of truth)
    if (options.saveOriginalFrames) {
        try {
            std::string origPath = (currentSession->outputPath / "processing_temp" /
                ("frame_" + std::to_string(frameIndex) + ".png")).string();

            if (!cv::imwrite(origPath, frame.colorImage)) {
                spdlog::warn("Failed to write original frame {} to {}", frameIndex, origPath);
                success = false;
            }
        } catch (const cv::Exception& e) {
            spdlog::error("Failed to save original image: {}", e.what());
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

            std::ofstream depthFile(depthPath, std::ios::binary);

            if (depthFile.is_open()) {
                int rows = frame.depthImage.rows;
                int cols = frame.depthImage.cols;

                // Write dimensions
                depthFile.write(reinterpret_cast<const char*>(&rows), sizeof(int));
                depthFile.write(reinterpret_cast<const char*>(&cols), sizeof(int));

                // Write raw depth data
                depthFile.write(reinterpret_cast<const char*>(frame.depthImage.data),
                              frame.depthImage.total() * frame.depthImage.elemSize());

                depthFile.close();

                // Only index frames that should be processed
                if (frameIndex % options.processEveryNFrames == 0) {
                    static std::mutex indexMutex;
                    std::lock_guard<std::mutex> lock(indexMutex);

                    std::string indexPath = (currentSession->outputPath / "process_frames.txt").string();
                    std::ofstream indexFile(indexPath, std::ios::app);

                    if (indexFile.is_open()) {
                        indexFile << frameIndex << std::endl;
                        indexFile.close();
                    }
                }
            } else {
                spdlog::warn("Failed to open depth file for writing: {}", depthPath);
                success = false;
            }
        } catch (const std::exception& e) {
            spdlog::error("Failed to save depth data: {}", e.what());
            success = false;
        }
    }

    // Update metadata
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