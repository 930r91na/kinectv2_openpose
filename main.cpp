#include "KinectDepthChecker.h"
#include "FrameRecorder.h"
#include "OpenPoseCapture.h"
#include "ConfigManager.h"
#include <iostream>
#include <filesystem>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

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
        } else if (arg == "--record") {
            isRecordMode = true;
        } else if ((arg == "--name" || arg == "--record" || arg == "--process") && i+1 < argc) {
            recordingName = argv[i+1];
            i++;
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

    // Load configuration
    ConfigManager config(configFile);
    std::string openPosePath = config.get<std::string>("openpose_path",
        "C:\\Users\\koqui\\OpenPose\\openpose\\bin\\OpenPoseDemo.exe");
    int netResolution = config.get<int>("net_resolution", 368);
    bool useMaximumAccuracy = config.get<bool>("use_maximum_accuracy", false);
    int confidenceThreshold = config.get<int>("keypoint_confidence_threshold", 40);
    int frameInterval = config.get<int>("process_every_n_frames", processingInterval);

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

            bool init = kinect.initialize();
            spdlog::info("Waiting for Kinect to fully initialize...");
            std::this_thread::sleep_for(std::chrono::seconds(3));

            if (!init) {
                spdlog::error("Failed to initialize Kinect");
                CoUninitialize();
                return -1;
            }

            // Initialize recorder
            std::string outputDir = "recordings/" + recordingName;
            FrameRecorder recorder(outputDir);

            // Set recorder options
            recorder.setProcessingInterval(frameInterval);
            recorder.setUseVideoCompression(useCompression);

            // Set up UI
            cv::namedWindow("Color Feed", cv::WINDOW_NORMAL);
            cv::namedWindow("Depth Feed", cv::WINDOW_NORMAL);

            // Main loop
            bool running = true;
            bool isRecording = false;
            int frameCount = 0;

            spdlog::info("Ready to record. Controls:");
            spdlog::info("  'r' - Start/stop recording");
            spdlog::info("  's' - Save the recording");
            spdlog::info("  'q' - Quit");

            auto lastFpsTime = std::chrono::high_resolution_clock::now();
            int displayFps = 0;

            while (running) {
                // Update Kinect data
                kinect.update();
                frameCount++;

                // Get images
                cv::Mat colorImg = kinect.getColorImage();
                cv::Mat depthImg = kinect.getDepthImage();
                cv::Mat depthViz;

                // Skip if frames are empty
                if (colorImg.empty() || depthImg.empty()) {
                    spdlog::warn("Empty frame detected");
                    continue;
                }

                // Add frame to recorder if recording
                if (isRecording) {
                    recorder.addFrame(colorImg, depthImg);
                }

                // Create visualization of the depth image
                cv::normalize(depthImg, depthViz, 0, 255, cv::NORM_MINMAX, CV_8UC1);
                cv::applyColorMap(depthViz, depthViz, cv::COLORMAP_JET);

                // Add recording indicator
                if (isRecording) {
                    cv::putText(colorImg, "RECORDING", cv::Point(30, 30),
                               cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
                    cv::circle(colorImg, cv::Point(20, 20), 10, cv::Scalar(0, 0, 255), -1);
                }

                // Calculate FPS
                auto currentTime = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastFpsTime);

                if (elapsed.count() >= 1) {
                    displayFps = frameCount;
                    frameCount = 0;
                    lastFpsTime = currentTime;
                }

                // Display FPS
                cv::putText(colorImg, "FPS: " + std::to_string(displayFps),
                           cv::Point(colorImg.cols - 150, 30), cv::FONT_HERSHEY_SIMPLEX,
                           1, cv::Scalar(255, 255, 255), 2);

                // Display images
                cv::resize(colorImg, colorImg, cv::Size(), 0.5, 0.5);
                cv::imshow("Color Feed", colorImg);
                cv::imshow("Depth Feed", depthViz);

                // Handle keyboard input
                int key = cv::waitKey(1);

                if (key == 'q' || key == 'Q') {
                    running = false;
                } else if (key == 'r' || key == 'R') {
                    if (isRecording) {
                        recorder.stopRecording();
                        isRecording = false;
                        spdlog::info("Recording stopped");
                    } else {
                        recorder.startRecording();
                        isRecording = true;
                        spdlog::info("Recording started");
                    }
                } else if (key == 's' || key == 'S') {
                    if (isRecording) {
                        recorder.stopRecording();
                        isRecording = false;
                    }

                    spdlog::info("Saving frames to disk...");
                    recorder.saveFramesToDisk();
                    spdlog::info("Recorded {} frames", recorder.getFrameCount());
                }
            }

            // Clean up
            if (isRecording) {
                recorder.stopRecording();
            }

            CoUninitialize();

        } catch (const std::exception& e) {
            spdlog::error("Unhandled exception: {}", e.what());
            return -1;
        }
    } else {
        // PROCESSING MODE
        spdlog::info("Processing recorded frames");

        std::string recordingDir = "recordings/" + recordingName;
        if (!fs::exists(recordingDir)) {
            spdlog::error("Recording directory not found: {}", recordingDir);
            return -1;
        }

        std::string outputDir = "processed/" + recordingName;

        try {
            // Initialize COM for Kinect
            if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) {
                spdlog::error("Failed to initialize COM");
                return -1;
            }

            // Initialize Kinect to get coordinate mapper
            KinectDepthChecker kinect;
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

            // Process recording using multi-threading
            bool success = openpose.processRecordingDirectory(
                recordingDir,
                kinect.getCoordinateMapper(),
                outputDir,
                numThreads
            );

            if (success) {
                spdlog::info("Successfully processed recording at {}", outputDir);
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