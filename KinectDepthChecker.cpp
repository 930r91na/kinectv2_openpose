#include "KinectDepthChecker.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <chrono>
#include <iostream>
#include <thread>

void KinectDepthChecker::setupLogger() {
    try {
        // Create console sink
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::info);

        // Create file sink
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("kinect_log.txt", true);
        file_sink->set_level(spdlog::level::trace);

        // Create logger with both sinks
        std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};
        logger = std::make_shared<spdlog::logger>("kinect", sinks.begin(), sinks.end());

        // Set pattern to include milliseconds
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
        logger->set_level(spdlog::level::trace);

        // Register it as the default logger
        spdlog::set_default_logger(logger);

        logger->info("Logger initialized");
    }
    catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
        throw;
    }
}

KinectDepthChecker::KinectDepthChecker() : sensor(nullptr), depthReader(nullptr), initialized(false) {
    setupLogger();
    logger->info("KinectDepthChecker initialized");
}

KinectDepthChecker::~KinectDepthChecker() {
    if (depthReader) {
        logger->debug("Releasing depth reader");
        depthReader->Release();
    }
    if (sensor) {
        logger->debug("Closing and releasing sensor");
        sensor->Close();
        sensor->Release();
    }
    logger->info("KinectDepthChecker destroyed");
}

bool KinectDepthChecker::initialize() {
    logger->info("Starting Kinect initialization");

    // Get default Kinect sensor
    HRESULT hr = GetDefaultKinectSensor(&sensor);
    if (FAILED(hr) || !sensor) {
        logger->error("Failed to get Kinect sensor, HRESULT: {:x}", hr);
        return false;
    }
    logger->info("Got default Kinect sensor");

    // Check sensor availability
    BOOLEAN isAvailable = FALSE;
    sensor->get_IsAvailable(&isAvailable);
    logger->info("Sensor availability check: {}", isAvailable ? "Available" : "Not Available");

    // Open sensor
    hr = sensor->Open();
    if (FAILED(hr)) {
        logger->error("Failed to open sensor, HRESULT: {:x}", hr);
        return false;
    }
    logger->info("Opened sensor");

    // Give the sensor some time to initialize
    logger->info("Waiting for sensor initialization...");
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Get depth frame source
    IDepthFrameSource* depthSource = nullptr;
    hr = sensor->get_DepthFrameSource(&depthSource);
    if (FAILED(hr) || !depthSource) {
        logger->error("Failed to get depth frame source, HRESULT: {:x}", hr);
        return false;
    }
    logger->info("Got depth frame source");

    // Get depth frame reader
    hr = depthSource->OpenReader(&depthReader);
    depthSource->Release();
    if (FAILED(hr) || !depthReader) {
        logger->error("Failed to get depth frame reader, HRESULT: {:x}", hr);
        return false;
    }
    logger->info("Got depth frame reader");

    initialized = true;
    logger->info("Kinect initialization completed successfully");
    return true;
}

void KinectDepthChecker::checkDepthFPS(int durationSeconds) const {
    if (!initialized) {
        logger->error("Device not initialized!");
        return;
    }

    int frameCount = 0;
    int failedFrames = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    logger->info("Starting FPS check for {} seconds", durationSeconds);
    logger->info("Capturing frames...");

    while (true) {
        IDepthFrame* frame = nullptr;
        HRESULT hr = depthReader->AcquireLatestFrame(&frame);

        if (SUCCEEDED(hr) && frame) {
            frameCount++;
            frame->Release();
            if (frameCount % 30 == 0) { // Log every 30 frames
                logger->debug("Captured {} frames", frameCount);
            }
        } else {
            failedFrames++;
            logger->trace("Frame acquisition failed, HRESULT: {:x}", hr);
        }

        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>
            (currentTime - startTime).count();

        if (elapsedSeconds >= durationSeconds) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    float fps = static_cast<float>(frameCount) / durationSeconds;

    logger->info("\nFPS Check Results:");
    logger->info("------------------");
    logger->info("Frames captured: {}", frameCount);
    logger->info("Failed frame reads: {}", failedFrames);
    logger->info("Average FPS: {:.1f}", fps);

    // Get depth camera resolution
    IDepthFrameSource* depthSource = nullptr;
    if (SUCCEEDED(sensor->get_DepthFrameSource(&depthSource))) {
        IFrameDescription* frameDescription = nullptr;
        if (SUCCEEDED(depthSource->get_FrameDescription(&frameDescription))) {
            int width = 0, height = 0;
            frameDescription->get_Width(&width);
            frameDescription->get_Height(&height);
            logger->info("Depth camera resolution: {}x{}", width, height);
            frameDescription->Release();
        }
        depthSource->Release();
    }

    if (fps < 1.0f) {
        logger->error("No meaningful frames captured - check Kinect connection");
    } else if (fps < 25.0f) {
        logger->warn("Frame rate below optimal (expected 30 FPS)");
    } else {
        logger->info("Frame rate within normal range");
    }
}