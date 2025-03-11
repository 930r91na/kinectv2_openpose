#include "Application.h"

#include <iostream>
#include <string>
#include <filesystem>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

// Function to display command-line help
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
    std::cout << "  --output path      Specify output directory for processing" << std::endl;
    std::cout << "  --help             Display this help" << std::endl;
}

int main(int argc, char** argv) {
    // Configure logging
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);


    try {
        // Parse command line arguments
        bool isRecordMode = true;
        std::string recordingName = "recording";
        std::string configFile = "config.ini";
        std::string outputDir;
        int processingInterval = 1;
        int numThreads = 4;
        bool useCompression = false;

        // Process command line arguments
        for (int i = 1; i < argc; i++) {
            if (std::string arg = argv[i]; arg == "--process") {
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
            } else if (arg == "--output" && i+1 < argc) {
                outputDir = argv[i+1];
                i++;
            } else if (arg == "--help") {
                displayHelp();
                return 0;
            }
        }

        // Initialize COM for Kinect
        if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) {
            spdlog::error("Failed to initialize COM");
            return -1;
        }

        // Create application instance
        Application app(configFile);

        // Initialize based on mode
        if (isRecordMode) {
            // Get configuration
            auto& config = app.getConfig();

            // Set command-line overrides
            if (processingInterval != 1) {
                config.setValue("process_every_n_frames", processingInterval);
            }
            if (useCompression) {
                config.setValue("use_video_compression", true);
            }

            // Initialize and run in record mode
            if (!app.initialize(Application::Mode::Record)) {
                spdlog::error("Failed to initialize application in record mode");
                CoUninitialize();
                return -1;
            }

            app.setSessionId(recordingName);

            spdlog::info("Application initialized in record mode");
            spdlog::info("Session ID set to: {}", recordingName);
            spdlog::info("Use 'r' to start/stop recording, 's' to save, 'q' to quit");

            // Run the application
            app.run();
        } else {
            // Process mode - determine paths
            std::string recordingsDir = app.getConfig().getValueOr<std::string>("recording_directory", "recordings");
            std::string recordingPath = recordingsDir + "/" + recordingName;

            // Check if recording exists
            if (!std::filesystem::exists(recordingPath)) {
                spdlog::error("Recording not found: {}", recordingPath);

                // List available recordings
                spdlog::info("Available recordings:");
                if (std::filesystem::exists(recordingsDir)) {
                    for (const auto& entry : std::filesystem::directory_iterator(recordingsDir)) {
                        if (entry.is_directory()) {
                            spdlog::info("  {}", entry.path().filename().string());
                        }
                    }
                }

                CoUninitialize();
                return -1;
            }

            // Configure processing
            auto& config = app.getConfig();
            config.setValue("processing_threads", numThreads);
            config.setValue("process_every_n_frames", processingInterval);

            // Initialize in process mode
            if (!app.initialize(Application::Mode::Process)) {
                spdlog::error("Failed to initialize application in process mode");
                CoUninitialize();
                return -1;
            }

            spdlog::info("Application initialized in process mode");
            spdlog::info("Processing recording: {}", recordingPath);

            // Start processing
            if (!app.startProcessing(recordingPath, outputDir)) {
                spdlog::error("Failed to start processing");
                CoUninitialize();
                return -1;
            }

            // Run until processing is complete
            app.run();

            // Get results
            auto result = app.getProcessingResult();
            if (result) {
                spdlog::info("Processing complete:");
                spdlog::info("  Frames processed: {}", result->framesProcessed);
                spdlog::info("  People detected: {}", result->peopleDetected);
                spdlog::info("  Processing time: {:.2f} seconds", result->processingTimeSeconds);
                spdlog::info("  Output saved to: {}", result->outputDirectory.string());
            } else {
                spdlog::error("Processing did not complete successfully");
            }
        }

        // Clean up COM
        CoUninitialize();

        return 0;
    } catch (const std::exception& e) {
        spdlog::error("Unhandled exception: {}", e.what());
        CoUninitialize();
        return -1;
    }
}