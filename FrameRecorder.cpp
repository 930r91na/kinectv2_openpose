#include "FrameRecorder.h"
#include "SkeletonExporter.h"
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <algorithm>

namespace fs = std::filesystem;

FrameRecorder::FrameRecorder(std::string outputDir)
    : outputDirectory(std::move(outputDir)), frameCounter(0) {

    if (!fs::exists(outputDirectory)) {
        fs::create_directories(outputDirectory);
    }
}

FrameRecorder::~FrameRecorder() {
    stopRecording();

    // Wait for any background thread to finish
    if (writerThread.joinable()) {
        shouldWrite = false;
        writerThread.join();
    }
}


void FrameRecorder::startRecording() {
    if (isRecording) {
        spdlog::warn("Recording already in progress");
        return;
    }

    // Make sure output directory exists
    if (!fs::exists(outputDirectory)) {
        fs::create_directories(outputDirectory);
        spdlog::info("Created output directory: {}", outputDirectory);
    }

    // Create a new recording directory with timestamp
    auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();

    // Use the output directory as the base path
    currentRecordingPath = outputDirectory;

    // Create absolute path for robustness
    fs::path absolutePath = fs::absolute(currentRecordingPath);
    currentRecordingPath = absolutePath.string(); // Store absolute path

    spdlog::info("Recording to directory: {}", currentRecordingPath);

    // Create all required directories
    fs::create_directories(currentRecordingPath);
    fs::create_directories(currentRecordingPath + "/depth_raw");

    if (!useVideoCompression) {
        fs::create_directories(currentRecordingPath + "/color");
    }

    // Verify directories were created successfully
    if (!fs::exists(currentRecordingPath)) {
        spdlog::error("Failed to create recording directory: {}", currentRecordingPath);
        return;
    }

    // Open metadata file
    std::string metadataPath = currentRecordingPath + "/metadata.txt";
    metadataStream.open(metadataPath);
    if (!metadataStream.is_open()) {
        spdlog::error("Failed to open metadata file at: {}", metadataPath);
        return;
    }

    metadataStream << "Recording started at: " << timestamp << std::endl;
    metadataStream << "Output directory: " << currentRecordingPath << std::endl;

    // Initialize video writer for color frames if using compression
    if (useVideoCompression) {
        // Use H.264 codec (or another appropriate codec)
        int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
        std::string videoPath = currentRecordingPath + "/color.mp4";
        colorVideoWriter.open(videoPath, fourcc, targetFps, cv::Size(1920, 1080));

        if (!colorVideoWriter.isOpened()) {
            spdlog::error("Failed to create video writer at {}, falling back to PNG frames", videoPath);
            useVideoCompression = false;
            fs::create_directories(currentRecordingPath + "/color");
        } else {
            spdlog::info("Created MP4 video writer with target FPS: {}", targetFps);
        }
    }

    // Reset frame counter and timestamps
    frameCounter = 0;
    recordingStartTime = std::chrono::high_resolution_clock::now();
    lastFrameTime = recordingStartTime;

    // Start processing thread
    isProcessing = true;
    shouldWrite = true;
    if (writerThread.joinable()) {
        writerThread.join(); // Ensure any previous thread is properly joined
    }
    writerThread = std::thread(&FrameRecorder::processFrameQueue, this);

    isRecording = true;
    spdlog::info("Recording started to {}", currentRecordingPath);
}

void FrameRecorder::stopRecording() {
    if (!isRecording) return;

    isRecording = false;

    // Signal processing thread to finish
    {
        std::unique_lock<std::mutex> lock(framesMutex);
        processingCv.notify_all();
    }

    // Wait for queue to empty
    {
        std::unique_lock<std::mutex> lock(framesMutex);
        processingCv.wait(lock, [this]() { return frameQueue.empty() || !isProcessing; });
    }

    isProcessing = false;
    processingCv.notify_all();

    if (writerThread.joinable()) {
        writerThread.join();
    }

    // Close metadata file and write final count
    if (metadataStream.is_open()) {
        auto recordingEndTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            recordingEndTime - recordingStartTime).count();

        double actualFps = static_cast<double>(frameCounter) / (duration / 1000.0);

        metadataStream << "Total frames: " << frameCounter << std::endl;
        metadataStream << "Recording duration (ms): " << duration << std::endl;
        metadataStream << "Actual FPS: " << actualFps << std::endl;
        metadataStream << "Target FPS: " << targetFps << std::endl;
        metadataStream.close();
    }

    // Close video writer
    if (colorVideoWriter.isOpened()) {
        colorVideoWriter.release();
    }

    // Save all the frame timestamps to a file for analysis
    if (!frameTimestamps.empty()) {
        std::ofstream timestampFile(currentRecordingPath + "/frame_timestamps.csv");
        timestampFile << "frame_index,timestamp,elapsed_ms,delta_ms" << std::endl;

        uint64_t startTs = frameTimestamps.front();
        uint64_t prevTs = startTs;

        for (size_t i = 0; i < frameTimestamps.size(); i++) {
            uint64_t ts = frameTimestamps[i];
            uint64_t elapsed = ts - startTs;
            uint64_t delta = ts - prevTs;

            timestampFile << i << "," << ts << "," << elapsed << "," << delta << std::endl;
            prevTs = ts;
        }

        timestampFile.close();
    }

    spdlog::info("Recording stopped. Captured {} frames", frameCounter);
}

void FrameRecorder::addFrame(const cv::Mat& colorImg, const cv::Mat& depthImg) {
    if (!isRecording) return;

    // Calculate time since last frame for FPS throttling
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto timeSinceLastFrame = std::chrono::duration_cast<std::chrono::milliseconds>(
        currentTime - lastFrameTime).count();

    // If we're trying to maintain a target FPS, throttle frame capture
    if (frameLimitingEnabled && targetFps > 0) {
        int minFrameInterval = 1000 / targetFps;
        if (timeSinceLastFrame < minFrameInterval) {
            // Skip this frame to maintain target FPS
            return;
        }
    }

    // Update last frame time
    lastFrameTime = currentTime;

    FramePair framePair;
    framePair.colorImage = colorImg.clone();
    framePair.depthImage = depthImg.clone();
    framePair.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    frameTimestamps.push_back(framePair.timestamp);

    {
        std::unique_lock<std::mutex> lock(framesMutex);

        // Wait if queue is full to prevent excessive memory usage
        processingCv.wait(lock, [this]() {
            return frameQueue.size() < MAX_QUEUE_SIZE || !isRecording;
        });

        if (!isRecording) return;

        frameQueue.push(std::move(framePair));
    }

    processingCv.notify_one();
}

void FrameRecorder::processFrameQueue() {
    spdlog::debug("Frame processing thread started");

    // Track issues for diagnostics
    int failedColorWrites = 0;
    int failedDepthWrites = 0;
    int processedFrames = 0;

    // Create performance metrics
    auto processingStartTime = std::chrono::high_resolution_clock::now();
    int lastLoggedFrameCount = 0;

    while (isProcessing) {
        FramePair frame;
        bool hasFrame = false;

        // Get a frame from the queue
        {
            std::unique_lock<std::mutex> lock(framesMutex);
            if (!frameQueue.empty()) {
                frame = std::move(frameQueue.front());
                frameQueue.pop();
                hasFrame = true;
                processingCv.notify_all(); // Notify if queue was full
            } else if (!isRecording) {
                // No more frames and recording stopped
                spdlog::debug("Queue empty and recording stopped, exiting processor thread");
                break;
            } else {
                // Wait for more frames
                processingCv.wait(lock);
                continue;
            }
        }

        if (hasFrame) {
            try {
                // Save frame to disk - use atomic increment for thread safety
                int currentFrame = frameCounter.fetch_add(1);
                processedFrames++;

                // Always save color frame
                if (useVideoCompression) {
                    if (colorVideoWriter.isOpened()) {
                        // Make sure image is BGR format for VideoWriter
                        cv::Mat bgrImage;
                        if (frame.colorImage.channels() == 4) {
                            cv::cvtColor(frame.colorImage, bgrImage, cv::COLOR_BGRA2BGR);
                        } else if (frame.colorImage.channels() == 3) {
                            bgrImage = frame.colorImage;
                        } else {
                            cv::cvtColor(frame.colorImage, bgrImage, cv::COLOR_GRAY2BGR);
                        }

                        // Add frame number overlay to the video
                        cv::putText(bgrImage, "Frame: " + std::to_string(currentFrame),
                                  cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                                  cv::Scalar(0, 255, 0), 2);

                        colorVideoWriter.write(bgrImage);
                    } else {
                        failedColorWrites++;
                        spdlog::warn("Video writer not opened, frame {} not saved", currentFrame);
                    }
                } else {
                    // Save as PNG if not using video compression
                    std::string colorPath = currentRecordingPath + "/color/frame_" +
                                          std::to_string(currentFrame) + ".png";
                    bool success = cv::imwrite(colorPath, frame.colorImage);
                    if (!success) {
                        failedColorWrites++;
                        spdlog::warn("Failed to write color frame {} to {}", currentFrame, colorPath);
                    }
                }

                    // Make sure depth image is valid
                    if (!frame.depthImage.empty() && frame.depthImage.type() == CV_16UC1) {
                        std::string depthRawPath = currentRecordingPath + "/depth_raw/frame_" +
                                                std::to_string(currentFrame) + ".bin";
                        std::ofstream depthFile(depthRawPath, std::ios::binary);

                        if (depthFile.is_open()) {
                            int rows = frame.depthImage.rows;
                            int cols = frame.depthImage.cols;
                            depthFile.write(reinterpret_cast<const char*>(&rows), sizeof(int));
                            depthFile.write(reinterpret_cast<const char*>(&cols), sizeof(int));

                            // Write raw depth data
                            depthFile.write(reinterpret_cast<const char*>(frame.depthImage.data),
                                          frame.depthImage.total() * frame.depthImage.elemSize());
                            depthFile.close();

                            // Save frame index for processing
                            {
                                // Use a critical section for the index file
                                static std::mutex indexFileMutex;
                                std::lock_guard<std::mutex> lock(indexFileMutex);

                                std::ofstream indexFile(currentRecordingPath + "/process_frames.txt",
                                                      std::ios::app);
                                indexFile << currentFrame << std::endl;
                            }
                        } else {
                            failedDepthWrites++;
                            spdlog::warn("Failed to open depth file for writing: {}", depthRawPath);
                        }
                    } else {
                        failedDepthWrites++;
                        spdlog::warn("Invalid depth image for frame {}", currentFrame);
                    }


                // Update metadata
                if (metadataStream.is_open()) {
                    static std::mutex metadataMutex;
                    std::lock_guard<std::mutex> lock(metadataMutex);

                    metadataStream << "Frame: " << currentFrame <<
                                  " Timestamp: " << frame.timestamp;
                    metadataStream << std::endl;
                }

                // Log progress and metrics occasionally
                if (processedFrames % 30 == 0 || processedFrames - lastLoggedFrameCount >= 30) {
                    auto currentTime = std::chrono::high_resolution_clock::now();
                    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                        currentTime - processingStartTime).count();
                    double fps = processedFrames * 1000.0 / elapsedMs;

                    spdlog::info("Processed {} frames (writing at {:.1f} FPS)",
                                processedFrames, fps);
                    lastLoggedFrameCount = processedFrames;
                }
            } catch (const std::exception& e) {
                spdlog::error("Error processing frame: {}", e.what());
            }
        }
    }

    // Log completion statistics
    spdlog::info("Frame processor finished. Processed {} frames ({} color failures, {} depth failures)",
                processedFrames, failedColorWrites, failedDepthWrites);
}

// Set the target FPS for recording
void FrameRecorder::setTargetFPS(int fps) {
    targetFps = fps;
    spdlog::info("Set target recording FPS to {}", targetFps);
}

// Enable or disable frame limiting based on target FPS
void FrameRecorder::setFrameLimiting(bool enabled) {
    frameLimitingEnabled = enabled;
    spdlog::info("Frame rate limiting set to: {}", enabled ? "enabled" : "disabled");
}

bool FrameRecorder::saveFramesToDisk() {
    // Make sure recording has stopped
    if (isRecording) {
        spdlog::warn("Cannot save frames while recording is in progress");
        return false;
    }

    // Make sure processing thread has completed
    if (isProcessing) {
        spdlog::info("Waiting for remaining frames to be processed...");

        // Wait for queue to empty
        {
            std::unique_lock<std::mutex> lock(framesMutex);
            processingCv.wait(lock, [this]() { return frameQueue.empty() || !isProcessing; });
        }

        // Give a bit more time for any pending file operations
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Check if path is valid
    if (currentRecordingPath.empty()) {
        spdlog::error("Recording path is empty. No recording has been started yet.");
        return false;
    }

    if (!fs::exists(currentRecordingPath)) {
        spdlog::error("Recording path does not exist: {}", currentRecordingPath);
        return false;
    }

    // Update metadata with final count if file is still open
    if (metadataStream.is_open()) {
        auto recordingEndTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            recordingEndTime - recordingStartTime).count();

        double actualFps = static_cast<double>(frameCounter) / (duration / 1000.0);

        metadataStream << "Total frames: " << frameCounter << std::endl;
        metadataStream << "Recording duration (ms): " << duration << std::endl;
        metadataStream << "Actual FPS: " << actualFps << std::endl;
        metadataStream << "Target FPS: " << targetFps << std::endl;
        metadataStream.close();
    }

    // Generate a summary file with frame information
    try {
        std::ofstream summaryFile(currentRecordingPath + "/recording_summary.txt");
        summaryFile << "Recording Summary" << std::endl;
        summaryFile << "----------------" << std::endl;
        summaryFile << "Total frames: " << frameCounter << std::endl;
        summaryFile << "Recording path: " << currentRecordingPath << std::endl;

        // Count files in directories to verify
        size_t colorFrames = 0;
        size_t depthFrames = 0;

        if (useVideoCompression && fs::exists(currentRecordingPath + "/color.mp4")) {
            summaryFile << "Color format: MP4 video" << std::endl;
            colorFrames = 1; // We have the video file
        } else if (fs::exists(currentRecordingPath + "/color")) {
            for (const auto& entry : fs::directory_iterator(currentRecordingPath + "/color")) {
                if (entry.path().extension() == ".png") {
                    colorFrames++;
                }
            }
            summaryFile << "Color format: PNG sequence" << std::endl;
            summaryFile << "Color frames: " << colorFrames << std::endl;
        }

        if (fs::exists(currentRecordingPath + "/depth_raw")) {
            for (const auto& entry : fs::directory_iterator(currentRecordingPath + "/depth_raw")) {
                if (entry.path().extension() == ".bin") {
                    depthFrames++;
                }
            }
            summaryFile << "Depth frames: " << depthFrames << std::endl;
        }

        // Calculate actual FPS from timestamps if available
        if (frameTimestamps.size() >= 2) {
            uint64_t startTime = frameTimestamps.front();
            uint64_t endTime = frameTimestamps.back();
            double durationSec = (endTime - startTime) / 1000000000.0; // Convert to seconds
            double calculatedFps = frameTimestamps.size() / durationSec;

            summaryFile << "Recording duration (sec): " << durationSec << std::endl;
            summaryFile << "Calculated FPS: " << calculatedFps << std::endl;

            // Also append to main log
            spdlog::info("Calculated actual recording FPS: {:.2f} over {:.2f} seconds",
                        calculatedFps, durationSec);
        }

        summaryFile.close();

        // Update frame counter based on actual files if necessary
        if (frameCounter == 0 && colorFrames > 0) {
            frameCounter = colorFrames;
        }
    } catch (const std::exception& e) {
        spdlog::warn("Failed to create summary file: {}", e.what());
    }

    // Create an MP4 from the PNG sequence if we didn't use video compression
    if (!useVideoCompression && fs::exists(currentRecordingPath + "/color")) {
    try {
        // Count PNG files and determine framerate
        std::vector<std::string> imageFiles;
        for (const auto& entry : fs::directory_iterator(currentRecordingPath + "/color")) {
            if (entry.path().extension() == ".png") {
                imageFiles.push_back(entry.path().string());
            }
        }

        // Sort files by frame number
        std::sort(imageFiles.begin(), imageFiles.end());

        if (!imageFiles.empty()) {
            // Determine the appropriate FPS based on actual recording data
            double fps = 5.0; // Default fallback FPS

            // Method 1: Calculate actual FPS from timestamps
            if (frameTimestamps.size() >= 2) {
                uint64_t startTime = frameTimestamps.front();
                uint64_t endTime = frameTimestamps.back();
                double durationSec = (endTime - startTime) / 1000000000.0;

                if (durationSec > 0) {
                    fps = frameTimestamps.size() / durationSec;
                    spdlog::info("Calculated FPS from timestamps: {:.2f}", fps);
                }
            }

            // Method 2: If timestamps aren't reliable, try to calculate from file timestamps
            if (fps <= 0.1 && imageFiles.size() >= 2) {
                auto firstTime = fs::last_write_time(imageFiles.front());
                auto lastTime = fs::last_write_time(imageFiles.back());

                // Convert to duration
                auto duration = lastTime - firstTime;
                auto durationSec = std::chrono::duration_cast<std::chrono::seconds>(duration).count();

                if (durationSec > 0) {
                    fps = imageFiles.size() / static_cast<double>(durationSec);
                    spdlog::info("Calculated FPS from file timestamps: {:.2f}", fps);
                }
            }

            // Method 3: Read from metadata if available
            if (fps <= 0.1) {
                std::string metadataPath = currentRecordingPath + "/metadata.txt";
                if (fs::exists(metadataPath)) {
                    std::ifstream metaFile(metadataPath);
                    std::string line;

                    while (std::getline(metaFile, line)) {
                        if (line.find("Actual FPS:") != std::string::npos) {
                            size_t pos = line.find(":");
                            if (pos != std::string::npos) {
                                std::string fpsStr = line.substr(pos + 1);
                                try {
                                    fps = std::stod(fpsStr);
                                    spdlog::info("Found FPS in metadata: {:.2f}", fps);
                                    break;
                                } catch (...) {
                                    // If parsing fails, continue with default
                                }
                            }
                        }
                    }
                }
            }

            // Ensure a reasonable FPS range
            fps = std::max(1.0, std::min(fps, 30.0));

            // Create video writer
            cv::VideoWriter videoWriter;
            std::string videoPath = currentRecordingPath + "/recording.mp4";

            // Read first image to get dimensions
            cv::Mat firstImage = cv::imread(imageFiles[0]);
            if (!firstImage.empty()) {
                int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                videoWriter.open(videoPath, fourcc, fps, firstImage.size());

                if (videoWriter.isOpened()) {
                    spdlog::info("Creating MP4 video from PNG sequence at {:.2f} FPS...", fps);

                    // Add each image to the video
                    for (const auto& imagePath : imageFiles) {
                        cv::Mat image = cv::imread(imagePath);
                        if (!image.empty()) {
                            videoWriter.write(image);
                        }
                    }

                    videoWriter.release();
                    spdlog::info("Created MP4 video at {}", videoPath);
                } else {
                    spdlog::error("Failed to create MP4 video writer");
                }
            }
        }
    } catch (const std::exception& e) {
        spdlog::error("Error creating MP4 from PNG sequence: {}", e.what());
    }
}
    spdlog::info("Frames saved to disk at {}", currentRecordingPath);
    spdlog::info("Recorded {} frames", frameCounter);
    return true;
}