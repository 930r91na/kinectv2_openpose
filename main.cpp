#include "KinectDepthChecker.h"
#include <iostream>
#include "spdlog/spdlog.h"

int main() {
    // Disable OpenCv logging
    setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    try {
        if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) {
            std::cout << "Failed to initialize COM" << std::endl;
            return -1;
        }

        std::cout << "Kinect V2 Depth Camera Checker" << std::endl;
        std::cout << "-----------------------------" << std::endl;

        KinectDepthChecker checker;

        if (!checker.initialize()) {
            spdlog::error("Failed to initialize Kinect");
            CoUninitialize();
            return -1;
        }

        checker.checkDepthFPS(5);

        // Main processing loop
        bool running = true;
        while (running) {
            checker.update();

            // Exit on 'q' press
            if (cv::waitKey(30) == 'q') {
                running = false;
            }
        }

        CoUninitialize();
    }
    catch (const std::exception& e) {
        spdlog::error("Unhandled exception: {}", e.what());
        return -1;
    }

    return 0;
}