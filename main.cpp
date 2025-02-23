#include "KinectDepthChecker.h"
#include <iostream>

#include "spdlog/spdlog.h"
#include "spdlog/details/os-inl.h"

int main() {
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

        checker.checkDepthFPS(10);

        CoUninitialize();
    }
    catch (const std::exception& e) {
        spdlog::error("Unhandled exception: {}", e.what());
        return -1;
    }

    return 0;
}
