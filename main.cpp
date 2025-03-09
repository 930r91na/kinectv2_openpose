#include "KinectDepthChecker.h"
#include "FrameRecorder.h"
#include "OpenPoseCapture.h"
#include "ConfigManager.h"
#include <iostream>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <chrono>

namespace fs = std::filesystem;

// Generate a unique session ID based on timestamp and name
std::string generateSessionId(const std::string& baseName) {
    auto now = std::chrono::system_clock::now();
    auto nowTime = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << baseName << "_" << std::put_time(std::localtime(&nowTime), "%Y%m%d_%H%M%S");
    return ss.str();
}

void displayHelp() {
    std::cout << "KinectV2 OpenPose Integrator" << std::endl;
    std::cout << "----------------------------" << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  --record [name]    Record mode (default)" << std::endl;
    std::cout << "  --process [name]   Process a recording" << std::endl;
    std::cout << "  --interval N       Process every Nth frame (default: 1)" << std::endl;
    std::cout << "  --threads N        Use N threads for processing (default: 4)" << std::endl;
    std::cout << "  --compress         Use video compression for color frames" << std::endl;
    std::cout << "  --config file.ini  Use specific config file (default: config.ini)" << std::endl;
    std::cout << "  --help             Display this help" << std::endl;
}

int main(int argc, char** argv) {
    // Configure logging
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    // Parse command line arguments
    bool isRecordMode = true;
    std::string recordingName = "recording";
    std::string configFile = "config.ini";
    int processingInterval = 1;
    int numThreads = 4;
    bool useCompression = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--process") {
            isRecordMode = false;
            // Get the session name if provided
            if (i+1 < argc && argv[i+1][0] != '-') {
                recordingName = argv[i+1];
                i++;
            }
        } else if (arg == "--record") {
            isRecordMode = true;
            // Get the session name if provided
            if (i+1 < argc && argv[i+1][0] != '-') {
                recordingName = argv[i+1];
                i++;
            }
        } else if (arg == "--interval" && i+1 < argc) {
            processingInterval = std::stoi(argv[i+1]);
            i++;
        } else if (arg == "--threads" && i+1 < argc) {
            numThreads = std::stoi(argv[i+1]);
            i++;
        } else if (arg == "--compress") {
            useCompression = true;
        } else if (arg == "--config" && i+1 < argc) {
            configFile = argv[i+1];
            i++;
        } else if (arg == "--help") {
            displayHelp();
            return 0;
        }
    }

    // Generate a unique session ID if we're recording
    std::string sessionId = isRecordMode ?
        generateSessionId(recordingName) : recordingName;

    // Display the session ID for clarity
    if (isRecordMode) {
        spdlog::info("Session ID: {}", sessionId);
    } else {
        spdlog::info("Processing session: {}", sessionId);
    }

    // Load configuration
    ConfigManager config(configFile);
    spdlog::info("Loaded configuration from {}", configFile);

    std::string openPosePath = config.get<std::string>("openpose_path",
        "C:\\Users\\koqui\\OpenPose\\openpose\\bin\\OpenPoseDemo.exe");
    int netResolution = config.get<int>("net_resolution", 368);
    bool useMaximumAccuracy = config.get<bool>("use_maximum_accuracy", false);
    int confidenceThreshold = config.get<int>("keypoint_confidence_threshold", 40);
    int frameInterval = config.get<int>("process_every_n_frames", processingInterval);

    // Get recording and output directories
    std::string recordingsBaseDir = config.get<std::string>("recording_directory", "recordings");
    std::string outputBaseDir = config.get<std::string>("output_directory", "processed");

    // Override config with command line if specified
    if (processingInterval != 1) {
        frameInterval = processingInterval;
    }

    if (isRecordMode) {
        // RECORDING MODE
        try {
            // Initialize COM for Kinect
            if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) {
                spdlog::error("Failed to initialize COM");
                return -1;
            }

            spdlog::info("Kinect V2 Frame Recorder");
            spdlog::info("----------------------");

            // Initialize Kinect
            KinectDepthChecker kinect;
            kinect.setShowWindows(false); // Disable default windows

            bool init = kinect.initialize();
            spdlog::info("Waiting for Kinect to fully initialize...");
            std::this_thread::sleep_for(std::chrono::seconds(3));

            if (!init) {
                spdlog::error("Failed to initialize Kinect");
                CoUninitialize();
                return -1;
            }

            // Make sure recording directory exists
            if (!fs::exists(recordingsBaseDir)) {
                fs::create_directories(recordingsBaseDir);
                spdlog::info("Created recordings directory: {}", recordingsBaseDir);
            }

            // Initialize recorder with full session path
            std::string sessionDir = recordingsBaseDir + "/" + sessionId;
            spdlog::info("Recording will be saved to: {}", sessionDir);

            FrameRecorder recorder(sessionDir);

            // Set recorder options
            recorder.setProcessingInterval(frameInterval);
            recorder.setUseVideoCompression(useCompression);

            // Set up a single control window
            cv::namedWindow("Kinect Recorder Control", cv::WINDOW_NORMAL);
            cv::resizeWindow("Kinect Recorder Control", 1280, 720);

            // Main loop
            bool running = true;
            bool isRecording = false;
            int frameCount = 0;

            // Default empty images for display
            cv::Mat controlImg = cv::Mat(720, 1280, CV_8UC3, cv::Scalar(40, 40, 40));
            cv::Mat colorSmall, depthSmall, skeletonSmall;

            // Display instructions on the control window
            cv::putText(controlImg, "KINECT RECORDER CONTROL",
                       cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
            cv::putText(controlImg, "Press 'r' to Start/Stop recording",
                       cv::Point(30, 120), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
            cv::putText(controlImg, "Press 's' to Save the recording",
                       cv::Point(30, 160), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
            cv::putText(controlImg, "Press 'q' to Quit",
                       cv::Point(30, 200), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
            cv::putText(controlImg, "Session: " + sessionId,
                       cv::Point(30, 260), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 1);

            // Show initial window
            cv::imshow("Kinect Recorder Control", controlImg);

            spdlog::info("Ready to record. Controls:");
            spdlog::info("  'r' - Start/stop recording");
            spdlog::info("  's' - Save the recording");
            spdlog::info("  'q' - Quit");

            auto lastFpsTime = std::chrono::high_resolution_clock::now();
            int displayFps = 0;
            int recordedFrames = 0;

            std::string statusText = "READY TO RECORD";

            while (running) {
                // Update Kinect data (without showing internal windows)
                kinect.update(false);
                frameCount++;

                // Get images
                cv::Mat colorImg = kinect.getColorImage();
                cv::Mat depthImg = kinect.getDepthImage();
                cv::Mat skeletonImg = kinect.getSkeletonImage();
                cv::Mat depthViz;

                // Skip if frames are empty
                if (colorImg.empty() || depthImg.empty()) {
                    spdlog::warn("Empty frame detected");
                    // Still show the window with status
                    controlImg = cv::Mat(720, 1280, CV_8UC3, cv::Scalar(40, 40, 40));
                    cv::putText(controlImg, "STATUS: WAITING FOR FRAMES",
                               cv::Point(700, 100), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
                    cv::putText(controlImg, "Press 'q' to Quit",
                               cv::Point(700, 240), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 1);
                    cv::imshow("Kinect Recorder Control", controlImg);

                    if (cv::waitKey(30) == 'q') {
                        running = false;
                    }
                    continue;
                }

                // Add frame to recorder if recording
                if (isRecording) {
                    recorder.addFrame(colorImg, depthImg);
                    recordedFrames = recorder.getFrameCount();
                }

                // Create visualization of the depth image
                cv::normalize(depthImg, depthViz, 0, 255, cv::NORM_MINMAX, CV_8UC1);
                cv::applyColorMap(depthViz, depthViz, cv::COLORMAP_JET);

                // Make sure all images are 3-channel BGR for display
                if (colorImg.channels() == 4) {
                    cv::cvtColor(colorImg, colorImg, cv::COLOR_BGRA2BGR);
                } else if (colorImg.channels() == 1) {
                    cv::cvtColor(colorImg, colorImg, cv::COLOR_GRAY2BGR);
                }

                if (depthViz.channels() == 1) {
                    cv::cvtColor(depthViz, depthViz, cv::COLOR_GRAY2BGR);
                } else if (depthViz.channels() == 4) {
                    cv::cvtColor(depthViz, depthViz, cv::COLOR_BGRA2BGR);
                }

                if (skeletonImg.channels() == 1) {
                    cv::cvtColor(skeletonImg, skeletonImg, cv::COLOR_GRAY2BGR);
                } else if (skeletonImg.channels() == 4) {
                    cv::cvtColor(skeletonImg, skeletonImg, cv::COLOR_BGRA2BGR);
                }

                // Resize images for display
                cv::resize(colorImg, colorSmall, cv::Size(640, 360));
                cv::resize(depthViz, depthSmall, cv::Size(320, 240));
                cv::resize(skeletonImg, skeletonSmall, cv::Size(320, 240));

                // Calculate FPS
                auto currentTime = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastFpsTime);
                if (elapsed.count() >= 1) {
                    displayFps = frameCount;
                    frameCount = 0;
                    lastFpsTime = currentTime;
                }

                // Update control display
                controlImg = cv::Mat(720, 1280, CV_8UC3, cv::Scalar(40, 40, 40));

                // Draw color feed at top - ensure ROI is within bounds
                if (colorSmall.rows > 0 && colorSmall.cols > 0) {
                    colorSmall.copyTo(controlImg(cv::Rect(20, 40,
                        std::min(colorSmall.cols, controlImg.cols - 20),
                        std::min(colorSmall.rows, controlImg.rows - 40))));
                }

                // Draw depth and skeleton below - ensure ROIs are within bounds
                if (depthSmall.rows > 0 && depthSmall.cols > 0) {
                    depthSmall.copyTo(controlImg(cv::Rect(20, 420,
                        std::min(depthSmall.cols, controlImg.cols - 20),
                        std::min(depthSmall.rows, controlImg.rows - 420))));
                }

                if (skeletonSmall.rows > 0 && skeletonSmall.cols > 0) {
                    skeletonSmall.copyTo(controlImg(cv::Rect(360, 420,
                        std::min(skeletonSmall.cols, controlImg.cols - 360),
                        std::min(skeletonSmall.rows, controlImg.rows - 420))));
                }

                // Status text
                cv::putText(controlImg, "STATUS: " + statusText,
                           cv::Point(700, 100), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                           isRecording ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255), 2);

                // Instructions
                cv::putText(controlImg, "Press 'r' to Start/Stop recording",
                           cv::Point(700, 160), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 1);
                cv::putText(controlImg, "Press 's' to Save the recording",
                           cv::Point(700, 200), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 1);
                cv::putText(controlImg, "Press 'q' to Quit",
                           cv::Point(700, 240), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 1);

                // Recording info
                cv::rectangle(controlImg, cv::Point(690, 270), cv::Point(1260, 380), cv::Scalar(60, 60, 60), -1);
                cv::putText(controlImg, "Session: " + sessionId,
                           cv::Point(700, 300), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 1);
                cv::putText(controlImg, "Frames: " + std::to_string(recordedFrames),
                           cv::Point(700, 330), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 1);
                cv::putText(controlImg, "FPS: " + std::to_string(displayFps),
                           cv::Point(700, 360), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 1);

                // Add recording indicator
                if (isRecording) {
                    cv::circle(controlImg, cv::Point(670, 100), 10, cv::Scalar(0, 0, 255), -1);
                }

                // Display the control window
                cv::imshow("Kinect Recorder Control", controlImg);

                // Handle keyboard input in a non-blocking way
                int key = cv::waitKey(1);

                if (key == 'q' || key == 'Q') {
                    statusText = "QUITTING...";
                    spdlog::info("Quitting application...");
                    cv::imshow("Kinect Recorder Control", controlImg);
                    cv::waitKey(1);

                    running = false;

                    // Stop recording if active before quitting
                    if (isRecording) {
                        spdlog::info("Stopping active recording...");
                        recorder.stopRecording();
                        isRecording = false;
                    }

                } else if (key == 'r' || key == 'R') {
                    if (isRecording) {
                        statusText = "STOPPING RECORDING...";
                        cv::imshow("Kinect Recorder Control", controlImg);
                        cv::waitKey(1);

                        spdlog::info("Stopping recording...");
                        recorder.stopRecording();
                        isRecording = false;
                        statusText = "READY";

                        spdlog::info("Recording stopped");
                    } else {
                        statusText = "STARTING RECORDING...";
                        cv::imshow("Kinect Recorder Control", controlImg);
                        cv::waitKey(1);

                        spdlog::info("Starting recording to {}", sessionDir);
                        recorder.startRecording();
                        isRecording = true;
                        statusText = "RECORDING";

                        spdlog::info("Recording started");
                    }
                } else if (key == 's' || key == 'S') {
                    if (isRecording) {
                        statusText = "STOPPING RECORDING...";
                        cv::imshow("Kinect Recorder Control", controlImg);
                        cv::waitKey(1);

                        spdlog::info("Stopping recording before save...");
                        recorder.stopRecording();
                        isRecording = false;
                    }

                    statusText = "SAVING RECORDING...";
                    cv::imshow("Kinect Recorder Control", controlImg);
                    cv::waitKey(1);

                    spdlog::info("Saving frames to disk at {}", sessionDir);

                    // Do saving in a separate thread to avoid UI freezing
                    std::thread saveThread([&]() {
                        bool saveResult = recorder.saveFramesToDisk();
                        if (saveResult) {
                            spdlog::info("Successfully saved {} frames", recorder.getFrameCount());
                            recordedFrames = recorder.getFrameCount();
                            statusText = "SAVED " + std::to_string(recordedFrames) + " FRAMES";
                        } else {
                            spdlog::error("Failed to save recording");
                            statusText = "SAVE FAILED";
                        }
                    });
                    saveThread.detach();
                }
            }

            // Clean up
            cv::destroyAllWindows();
            CoUninitialize();

        } catch (const std::exception& e) {
            spdlog::error("Unhandled exception: {}", e.what());
            return -1;
        }
    } else {
        // PROCESSING MODE - Rest of code remains the same
        // [Processing code here - no changes needed]
        spdlog::info("Processing recorded frames");

        // Verify the full path for the recording directory
        std::string recordingDir = recordingsBaseDir + "/" + sessionId;
        if (!fs::exists(recordingDir)) {
            spdlog::error("Recording directory not found: {}", recordingDir);
            spdlog::info("Available recordings:");

            // List available recordings
            if (fs::exists(recordingsBaseDir)) {
                for (const auto& entry : fs::directory_iterator(recordingsBaseDir)) {
                    if (entry.is_directory()) {
                        spdlog::info("  {}", entry.path().filename().string());
                    }
                }
            }

            return -1;
        }

        std::string outputDir = outputBaseDir + "/" + sessionId;
        spdlog::info("Will process {} using {} threads", recordingDir, numThreads);
        spdlog::info("Output will be saved to {}", outputDir);

        try {
            // Initialize COM for Kinect
            if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) {
                spdlog::error("Failed to initialize COM");
                return -1;
            }

            // Initialize Kinect to get coordinate mapper
            KinectDepthChecker kinect;
            kinect.setShowWindows(false); // No need for visualization

            if (!kinect.initialize()) {
                spdlog::error("Failed to initialize Kinect");
                CoUninitialize();
                return -1;
            }

            // Initialize OpenPose
            OpenPoseCapture openpose(openPosePath);
            openpose.setNetResolution(netResolution);
            openpose.setMaximumAccuracy(useMaximumAccuracy);
            openpose.setKeypointConfidenceThreshold(confidenceThreshold);

            if (!openpose.initialize()) {
                spdlog::error("Failed to initialize OpenPose");
                CoUninitialize();
                return -1;
            }

            // Create output directory if it doesn't exist
            if (!fs::exists(outputDir)) {
                fs::create_directories(outputDir);
            }

            spdlog::info("Starting processing...");

            // Process recording using multi-threading
            bool success = openpose.processRecordingDirectory(
                recordingDir,
                kinect.getCoordinateMapper(),
                outputDir,
                numThreads
            );

            if (success) {
                spdlog::info("Successfully processed recording");
                spdlog::info("Results saved to {}", outputDir);
            } else {
                spdlog::error("Failed to process recording");
            }

            CoUninitialize();

        } catch (const std::exception& e) {
            spdlog::error("Unhandled exception during processing: {}", e.what());
            return -1;
        }
    }

    return 0;
}