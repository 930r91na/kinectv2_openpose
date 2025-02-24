#pragma once
#define SPDLOG_HEADER_ONLY
#include <Kinect.h>
#include <spdlog/spdlog.h>
#include <memory>
#include <opencv2/opencv.hpp>

// Safe release for interfaces
template<class Interface>
inline void SafeRelease(Interface*& pInterfaceToRelease) {
    if (pInterfaceToRelease != nullptr) {
        pInterfaceToRelease->Release();
        pInterfaceToRelease = NULL;
    }
}

class KinectDepthChecker {
    static constexpr int cDepthWidth = 512;
    static constexpr int cDepthHeight = 424;

    IKinectSensor* m_pKinectSensor;
    IDepthFrameReader* m_pDepthFrameReader;
    ICoordinateMapper* m_pCoordinateMapper;
    IBodyFrameReader* m_pBodyFrameReader;
    IBodyIndexFrameReader* m_pBodyIndexFrameReader;
    bool initialized;
    std::shared_ptr<spdlog::logger> logger;

    void setupLogger();

    // Mat objects for visualization
    cv::Mat skeletonImg;
    cv::Mat depthImg;

public:
    KinectDepthChecker();
    ~KinectDepthChecker();

    // Delete copy constructor and assignment operator
    KinectDepthChecker(const KinectDepthChecker&) = delete;
    KinectDepthChecker& operator=(const KinectDepthChecker&) = delete;

    bool initialize();
    void update();
    void checkDepthFPS(int durationSeconds = 10) const;

private:
    // Functions to process skeleton
    void ProcessBody(int nBodyCount, IBody** ppBodies);
    void DrawBone(const Joint* pJoints, const DepthSpacePoint* depthSpacePosition, JointType joint0, JointType joint1);
    void DrawHandState(const DepthSpacePoint depthSpacePosition, HandState handState);
};