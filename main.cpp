#include "KinectDepthChecker.h"
#include "OpenPoseCapture.h"
#include <iostream>
#include <filesystem>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

int main() {
    // Configure logging
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    // Disable OpenCV logging
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    try {
        // Initialize COM for Kinect
        if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) {
            spdlog::error("Failed to initialize COM");
            return -1;
        }

        spdlog::info("Kinect V2 + OpenPose 3D Skeleton Capture");
        spdlog::info("----------------------------------------");

        // Create output directory for 3D data
        std::string outputDir = "output_3d";
        if (!fs::exists(outputDir)) {
            fs::create_directory(outputDir);
        }

        // Initialize Kinect
        KinectDepthChecker kinect;
        if (!kinect.initialize()) {
            spdlog::error("Failed to initialize Kinect");
            CoUninitialize();
            return -1;
        }

        // Check Kinect FPS
        kinect.checkDepthFPS(3);

        // Initialize OpenPose with path to executable
        // Update this path to match your OpenPose installation
        OpenPoseCapture openpose("bin\\OpenPoseDemo.exe");

        // Configure OpenPose settings
        openpose.setNetResolution(368);       // Default balanced resolution
        openpose.setMaximumAccuracy(false);   // Set to true for maximum accuracy (slower)
        openpose.setKeypointConfidenceThreshold(40); // 40% confidence threshold

        if (!openpose.initialize()) {
            spdlog::error("Failed to initialize OpenPose");
            CoUninitialize();
            return -1;
        }

        // Processing parameters
        int frameCount = 0;
        int processedFrames = 0;
        int processingInterval = 15; // Process every 15th frame for performance
        auto lastProcessTime = std::chrono::high_resolution_clock::now();

        // Main processing loop
        bool running = true;
        spdlog::info("Starting main loop. Press:");
        spdlog::info("  'q' to exit");
        spdlog::info("  's' to save current frame as 3D skeleton");
        spdlog::info("  'p' to process current frame with OpenPose");

        while (running) {
            // Update Kinect - gets new depth and color frames
            kinect.update();
            frameCount++;

            // Get color and depth images
            cv::Mat colorImg = kinect.getColorImage();
            cv::Mat depthImg = kinect.getDepthImage();

            // Skip processing if either image is empty
            if (colorImg.empty() || depthImg.empty()) {
                spdlog::warn("Empty frame detected, skipping frame {}", frameCount);
                continue;
            }

            // Process at regular intervals or when triggered
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                currentTime - lastProcessTime).count();

            // Check if it's time to process a frame
            bool shouldProcessFrame = (frameCount % processingInterval == 0) && (elapsed > 500);

            // Handle keyboard input
            int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q') {
                running = false;
            } else if (key == 'p' || key == 'P') {
                // Force processing on 'p' key
                shouldProcessFrame = true;
            } else if (key == 's' || key == 'S') {
                // Save the current 3D skeleton on 's' key
                shouldProcessFrame = true;
                // We'll save the result below after processing
            }

            // Process frame if needed
            if (shouldProcessFrame) {
                spdlog::info("Processing frame {} with OpenPose", frameCount);
                std::vector<Person3D> people3D;

                // Process the frame with OpenPose and get 3D data
                if (openpose.processFrame(colorImg, depthImg, kinect.getCoordinateMapper(), people3D)) {
                    processedFrames++;
                    lastProcessTime = currentTime;

                    // Log the detected people and keypoints
                    spdlog::info("Detected {} people in the frame", people3D.size());

                    // For each person, log some of their 3D keypoints
                    for (size_t personIdx = 0; personIdx < people3D.size(); personIdx++) {
                        const auto& person = people3D[personIdx];
                        spdlog::info("Person {}: {} keypoints", personIdx, person.keypoints.size());

                        // Log a few key joints with their 3D coordinates
                        const std::vector<std::pair<int, std::string>> keyJoints = {
                            {0, "Nose"}, {1, "Neck"}, {8, "MidHip"},
                            {4, "RWrist"}, {7, "LWrist"}
                        };

                        for (const auto& [idx, name] : keyJoints) {
                            if (idx < person.keypoints.size()) {
                                const auto& kp = person.keypoints[idx];
                                spdlog::info("  Joint {}: ({:.3f}, {:.3f}, {:.3f}m) conf: {:.2f}",
                                             name, kp.x, kp.y, kp.z, kp.confidence);
                            }
                        }

                        // Visualize the 3D skeleton
                        cv::Mat visualImg = colorImg.clone();
                        openpose.visualize3DSkeleton(visualImg, people3D);

                        // Resize for display if needed
                        cv::Mat displayImg;
                        cv::resize(visualImg, displayImg, cv::Size(), 0.5, 0.5);
                        cv::imshow("3D Skeleton", displayImg);

                        // Save if 's' was pressed
                        if (key == 's' || key == 'S') {
                            std::string outputPath = outputDir + "/skeleton3d_" +
                                                  std::to_string(frameCount) + ".json";
                            if (openpose.save3DSkeletonToJson(people3D, outputPath)) {
                                spdlog::info("Saved 3D skeleton to {}", outputPath);
                            }
                        }
                    }
                } else {
                    spdlog::warn("No people detected or processing failed");
                }
            }
        }

        CoUninitialize();
        spdlog::info("Processed {} frames out of {} total frames", processedFrames, frameCount);
        spdlog::info("Average processing rate: {:.2f}%",
                     (processedFrames > 0) ? (100.0f * processedFrames / frameCount) : 0.0f);
    }
    catch (const std::exception& e) {
        spdlog::error("Unhandled exception: {}", e.what());
        return -1;
    }

    return 0;
}