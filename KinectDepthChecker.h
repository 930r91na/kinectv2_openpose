#pragma once
#define SPDLOG_HEADER_ONLY
#include <Kinect.h>
#include <spdlog/spdlog.h>
#include <memory>

class KinectDepthChecker {
private:
    IKinectSensor* sensor;
    IDepthFrameReader* depthReader;
    bool initialized;
    std::shared_ptr<spdlog::logger> logger;

    void setupLogger();

public:
    KinectDepthChecker();
    ~KinectDepthChecker();

    // Delete copy constructor and assignment operator
    KinectDepthChecker(const KinectDepthChecker&) = delete;
    KinectDepthChecker& operator=(const KinectDepthChecker&) = delete;

    bool initialize();
    void checkDepthFPS(int durationSeconds = 10) const;
};