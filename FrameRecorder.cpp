#include "FrameRecorder.h"
#include "SkeletonExporter.h"
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <chrono>

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
    if (isRecording) return;

    // Create a new recording directory with timestamp
    auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    currentRecordingPath = outputDirectory + "/recording_" + std::to_string(timestamp);

    // Create subdirectories - ensure we use absolute paths for robustness
    fs::path absolutePath = fs::absolute(currentRecordingPath);
    currentRecordingPath = absolutePath.string(); // Store absolute path

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

    // Initialize video writer for color frames if using compression
    if (useVideoCompression) {
        // Use H.264 codec (or another appropriate codec)
        int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
        std::string videoPath = currentRecordingPath + "/color.mp4";
        colorVideoWriter.open(videoPath, fourcc, 30.0, cv::Size(1920, 1080));

        if (!colorVideoWriter.isOpened()) {
            spdlog::error("Failed to create video writer at {}, falling back to PNG frames", videoPath);
            useVideoCompression = false;
            fs::create_directories(currentRecordingPath + "/color");
        }
    }

    // Reset frame counter
    frameCounter = 0;

    // Start processing thread
    isProcessing = true;
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
        metadataStream << "Total frames: " << frameCounter << std::endl;
        metadataStream.close();
    }

    // Close video writer
    if (colorVideoWriter.isOpened()) {
        colorVideoWriter.release();
    }

    spdlog::info("Recording stopped. Captured {} frames", frameCounter);
}

void FrameRecorder::addFrame(const cv::Mat& colorImg, const cv::Mat& depthImg) {
    if (!isRecording) return;

    FramePair framePair;
    framePair.colorImage = colorImg.clone();
    framePair.depthImage = depthImg.clone();
    framePair.timestamp = std::chrono::system_clock::now().time_since_epoch().count();

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

                // Process only every N frames to save storage/CPU
                bool shouldProcessDepth = (currentFrame % processingInterval == 0);

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

                // Save raw depth data for selected frames
                if (shouldProcessDepth) {
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
                }

                // Update metadata
                if (metadataStream.is_open()) {
                    static std::mutex metadataMutex;
                    std::lock_guard<std::mutex> lock(metadataMutex);

                    metadataStream << "Frame: " << currentFrame <<
                                  " Timestamp: " << frame.timestamp;
                    if (shouldProcessDepth) {
                        metadataStream << " HasDepth: yes";
                    }
                    metadataStream << std::endl;
                }

                // Log progress occasionally
                if (processedFrames % 100 == 0) {
                    spdlog::info("Processed {} frames", processedFrames);
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
    if (currentRecordingPath.empty() || !fs::exists(currentRecordingPath)) {
        spdlog::error("Recording path is invalid or doesn't exist: {}",
                     currentRecordingPath.empty() ? "[empty]" : currentRecordingPath);
        return false;
    }

    // Update metadata with final count if file is still open
    if (metadataStream.is_open()) {
        metadataStream << "Total frames: " << frameCounter << std::endl;
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

        summaryFile.close();

        // Update frame counter based on actual files if necessary
        if (frameCounter == 0 && colorFrames > 0) {
            frameCounter = colorFrames;
        }
    } catch (const std::exception& e) {
        spdlog::warn("Failed to create summary file: {}", e.what());
    }

    spdlog::info("Frames saved to disk at {}", currentRecordingPath);
    spdlog::info("Recorded {} frames", frameCounter);
    return true;
}

bool FrameRecorder::loadRecordedFrames(const std::string& directory, std::vector<FramePair>& outFrames) {
    if (!fs::exists(directory)) {
        spdlog::error("Directory does not exist: {}", directory);
        return false;
    }

    bool hasVideoFile = fs::exists(directory + "/color.mp4");
    bool hasColorDir = fs::exists(directory + "/color");
    bool hasDepthDir = fs::exists(directory + "/depth_raw");
    bool hasProcessList = fs::exists(directory + "/process_frames.txt");

    if ((!hasVideoFile && !hasColorDir) || !hasDepthDir) {
        spdlog::error("Missing required directories in {}", directory);
        return false;
    }

    outFrames.clear();

    // Load frame indices to process
    std::vector<int> framesToProcess;
    if (hasProcessList) {
        std::ifstream indexFile(directory + "/process_frames.txt");
        int frameIdx;
        while (indexFile >> frameIdx) {
            framesToProcess.push_back(frameIdx);
        }
        spdlog::info("Found {} frames to process from index file", framesToProcess.size());
    } else {
        // If no process list, count depth files
        for (const auto& entry : fs::directory_iterator(directory + "/depth_raw")) {
            if (entry.path().extension() == ".bin") {
                std::string filename = entry.path().filename().string();
                // Extract frame number from format "frame_X.bin"
                size_t underscore = filename.find('_');
                size_t dot = filename.find('.');
                if (underscore != std::string::npos && dot != std::string::npos) {
                    int frameIdx = std::stoi(filename.substr(underscore + 1, dot - underscore - 1));
                    framesToProcess.push_back(frameIdx);
                }
            }
        }
        std::sort(framesToProcess.begin(), framesToProcess.end());
        spdlog::info("Found {} depth frame files", framesToProcess.size());
    }

    if (framesToProcess.empty()) {
        spdlog::error("No frames found to process in {}", directory);
        return false;
    }

    outFrames.reserve(framesToProcess.size());

    // Open video capture if using video format
    cv::VideoCapture videoCapture;
    if (hasVideoFile) {
        videoCapture.open(directory + "/color.mp4");
        if (!videoCapture.isOpened()) {
            spdlog::error("Failed to open color video file");
            return false;
        }
    }

    for (int frameIdx : framesToProcess) {
        FramePair framePair;
        framePair.timestamp = frameIdx; // Use frame index as timestamp if real one not available

        // Load color image
        if (hasVideoFile) {
            // Seek to the right frame in the video
            videoCapture.set(cv::CAP_PROP_POS_FRAMES, frameIdx);
            if (!videoCapture.read(framePair.colorImage)) {
                spdlog::warn("Failed to read color frame {} from video", frameIdx);
                continue;
            }
        } else {
            // Load from PNG
            std::string colorPath = directory + "/color/frame_" + std::to_string(frameIdx) + ".png";
            framePair.colorImage = cv::imread(colorPath);
            if (framePair.colorImage.empty()) {
                spdlog::warn("Failed to load color frame {}", frameIdx);
                continue;
            }
        }

        // Load raw depth data
        std::string depthRawPath = directory + "/depth_raw/frame_" + std::to_string(frameIdx) + ".bin";
        std::ifstream depthFile(depthRawPath, std::ios::binary);

        if (depthFile.is_open()) {
            // Read dimensions
            int rows, cols;
            depthFile.read(reinterpret_cast<char*>(&rows), sizeof(int));
            depthFile.read(reinterpret_cast<char*>(&cols), sizeof(int));

            // Allocate matrix
            framePair.depthImage = cv::Mat(rows, cols, CV_16UC1);

            // Read data
            depthFile.read(reinterpret_cast<char*>(framePair.depthImage.data),
                          framePair.depthImage.total() * framePair.depthImage.elemSize());
            depthFile.close();
        } else {
            spdlog::warn("Failed to load raw depth data for frame {}", frameIdx);
            continue;
        }

        outFrames.push_back(std::move(framePair));

        // Log progress every 100 frames
        if (outFrames.size() % 100 == 0) {
            spdlog::info("Loaded {}/{} frames", outFrames.size(), framesToProcess.size());
        }
    }

    if (hasVideoFile) {
        videoCapture.release();
    }

    spdlog::info("Successfully loaded {} frames", outFrames.size());
    return !outFrames.empty();
}