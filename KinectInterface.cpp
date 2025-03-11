#include "KinectInterface.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <thread>
#include <algorithm>

// Safe release template function for COM interfaces
template<class Interface>
inline void SafeRelease(Interface*& pInterfaceToRelease) {
    if (pInterfaceToRelease != nullptr) {
        pInterfaceToRelease->Release();
        pInterfaceToRelease = nullptr;
    }
}

// PIMPL implementation class
class KinectInterfaceImpl {
public:
    KinectInterfaceImpl() 
        : kinectSensor(nullptr),
          depthFrameReader(nullptr),
          colorFrameReader(nullptr),
          bodyFrameReader(nullptr),
          bodyIndexFrameReader(nullptr),
          coordinateMapper(nullptr),
          initialized(false),
          frameCount(0) {
        
        // Initialize frame buffers with proper sizes
        skeletonImage = cv::Mat(KinectInterface::Resolution::DepthHeight, 
                               KinectInterface::Resolution::DepthWidth, 
                               CV_8UC3, cv::Scalar(0, 0, 0));
                               
        depthImage = cv::Mat(KinectInterface::Resolution::DepthHeight, 
                            KinectInterface::Resolution::DepthWidth, 
                            CV_8UC1);
                            
        colorImage = cv::Mat(KinectInterface::Resolution::ColorHeight, 
                            KinectInterface::Resolution::ColorWidth, 
                            CV_8UC4);
                            
        rawDepthImage = cv::Mat(KinectInterface::Resolution::DepthHeight, 
                               KinectInterface::Resolution::DepthWidth, 
                               CV_16UC1);
    }

    ~KinectInterfaceImpl() {
        // Release all Kinect resources
        SafeRelease(depthFrameReader);
        SafeRelease(bodyFrameReader);
        SafeRelease(bodyIndexFrameReader);
        SafeRelease(coordinateMapper);
        SafeRelease(colorFrameReader);

        if (kinectSensor) {
            kinectSensor->Close();
            SafeRelease(kinectSensor);
        }
        
        spdlog::info("Kinect resources released");
    }

    bool initialize() {
        spdlog::info("Starting Kinect initialization");

        HRESULT hr = GetDefaultKinectSensor(&kinectSensor);
        if (FAILED(hr) || !kinectSensor) {
            spdlog::error("Failed to get Kinect sensor");
            return false;
        }

        hr = kinectSensor->Open();
        if (FAILED(hr)) {
            spdlog::error("Failed to open sensor");
            return false;
        }

        // Get the coordinate mapper
        hr = kinectSensor->get_CoordinateMapper(&coordinateMapper);
        if (FAILED(hr)) {
            spdlog::error("Failed to get coordinate mapper");
            return false;
        }

        // Get depth reader
        IDepthFrameSource* depthFrameSource = nullptr;
        hr = kinectSensor->get_DepthFrameSource(&depthFrameSource);
        if (SUCCEEDED(hr)) {
            hr = depthFrameSource->OpenReader(&depthFrameReader);
            SafeRelease(depthFrameSource);
            if (FAILED(hr)) {
                spdlog::error("Failed to create depth frame reader");
            }
        }

        // Get color reader
        IColorFrameSource* colorFrameSource = nullptr;
        hr = kinectSensor->get_ColorFrameSource(&colorFrameSource);
        if (SUCCEEDED(hr)) {
            hr = colorFrameSource->OpenReader(&colorFrameReader);
            if (FAILED(hr)) {
                spdlog::error("Failed to create color frame reader");
            } else {
                spdlog::info("Color frame reader created successfully");
            }
            SafeRelease(colorFrameSource);
        }

        // Get body reader
        IBodyFrameSource* bodyFrameSource = nullptr;
        hr = kinectSensor->get_BodyFrameSource(&bodyFrameSource);
        if (SUCCEEDED(hr)) {
            hr = bodyFrameSource->OpenReader(&bodyFrameReader);
            SafeRelease(bodyFrameSource);
            if (FAILED(hr)) {
                spdlog::error("Failed to create body frame reader");
            }
        }

        // Get body index reader
        IBodyIndexFrameSource* bodyIndexFrameSource = nullptr;
        hr = kinectSensor->get_BodyIndexFrameSource(&bodyIndexFrameSource);
        if (SUCCEEDED(hr)) {
            hr = bodyIndexFrameSource->OpenReader(&bodyIndexFrameReader);
            SafeRelease(bodyIndexFrameSource);
            if (FAILED(hr)) {
                spdlog::error("Failed to create body index frame reader");
            }
        }

        // Check if we have at least the essential readers
        if (!depthFrameReader || !colorFrameReader) {
            spdlog::error("Essential frame readers not available");
            return false;
        }

        initialized = true;
        spdlog::info("Kinect initialization completed successfully");
        return true;
    }

    bool update() {
        if (!initialized) {
            spdlog::warn("Attempted to update uninitialized Kinect");
            return false;
        }

        frameCount++;
        bool gotAnyFrame = false;

        // Clear skeleton image for drawing
        skeletonImage = cv::Mat::zeros(KinectInterface::Resolution::DepthHeight, 
                                      KinectInterface::Resolution::DepthWidth, 
                                      CV_8UC3);

        // Process depth frame
        if (updateDepthFrame()) {
            gotAnyFrame = true;
        }

        // Process color frame
        if (updateColorFrame()) {
            gotAnyFrame = true;
        }

        // Process body frame
        if (updateBodyFrame()) {
            gotAnyFrame = true;
        }

        // Trigger callback if set
        if (gotAnyFrame && frameReadyCallback) {
            frameReadyCallback();
        }

        return gotAnyFrame;
    }

    bool updateDepthFrame() {
        if (!depthFrameReader) return false;

        IDepthFrame* depthFrame = nullptr;
        HRESULT hr = depthFrameReader->AcquireLatestFrame(&depthFrame);
        
        if (SUCCEEDED(hr) && depthFrame) {
            // Get depth data
            UINT16* depthBuffer = new UINT16[KinectInterface::Resolution::DepthWidth * 
                                            KinectInterface::Resolution::DepthHeight];
            
            hr = depthFrame->CopyFrameDataToArray(
                KinectInterface::Resolution::DepthWidth * KinectInterface::Resolution::DepthHeight, 
                depthBuffer);
            
            if (SUCCEEDED(hr)) {
                // Store the raw depth data
                rawDepthImage = cv::Mat(KinectInterface::Resolution::DepthHeight, 
                                       KinectInterface::Resolution::DepthWidth, 
                                       CV_16UC1, 
                                       depthBuffer).clone();

                // Create a visualization of the depth data
                cv::Mat tempDepth(KinectInterface::Resolution::DepthHeight, 
                                 KinectInterface::Resolution::DepthWidth, 
                                 CV_16UC1, 
                                 depthBuffer);
                                 
                cv::normalize(tempDepth, depthImage, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            }
            
            delete[] depthBuffer;
            SafeRelease(depthFrame);
            return SUCCEEDED(hr);
        }
        
        SafeRelease(depthFrame);
        return false;
    }

    bool updateColorFrame() {
        if (!colorFrameReader) return false;

        IColorFrame* colorFrame = nullptr;
        HRESULT hr = colorFrameReader->AcquireLatestFrame(&colorFrame);
        
        if (SUCCEEDED(hr) && colorFrame) {
            // Get color frame description
            IFrameDescription* colorFrameDescription = nullptr;
            hr = colorFrame->get_FrameDescription(&colorFrameDescription);

            if (SUCCEEDED(hr)) {
                int width = 0, height = 0;
                colorFrameDescription->get_Width(&width);
                colorFrameDescription->get_Height(&height);

                // Make sure our Mat is the right size
                if (colorImage.cols != width || colorImage.rows != height) {
                    colorImage = cv::Mat(height, width, CV_8UC4);
                }

                // Copy color data to our Mat
                hr = colorFrame->CopyConvertedFrameDataToArray(
                    width * height * 4,  // 4 bytes per pixel (BGRA)
                    reinterpret_cast<BYTE*>(colorImage.data),
                    ColorImageFormat_Bgra
                );

                SafeRelease(colorFrameDescription);
            }
            
            SafeRelease(colorFrame);
            return SUCCEEDED(hr);
        }
        
        SafeRelease(colorFrame);
        return false;
    }

    bool updateBodyFrame() {
        if (!bodyFrameReader) return false;

        IBodyFrame* bodyFrame = nullptr;
        HRESULT hr = bodyFrameReader->AcquireLatestFrame(&bodyFrame);
        
        if (SUCCEEDED(hr) && bodyFrame) {
            // Clear previous body data
            trackedBodies.clear();
            
            // Get body data
            IBody* bodies[BODY_COUNT] = { nullptr };
            hr = bodyFrame->GetAndRefreshBodyData(_countof(bodies), bodies);
            
            if (SUCCEEDED(hr)) {
                processBodyData(bodies);
            }
            
            // Release body instances
            for (auto & body : bodies) {
                SafeRelease(body);
            }
            
            SafeRelease(bodyFrame);
            return SUCCEEDED(hr);
        }
        
        SafeRelease(bodyFrame);
        return false;
    }

    void processBodyData(IBody** bodies) {
        for (int i = 0; i < BODY_COUNT; ++i) {
            IBody* body = bodies[i];
            if (!body) continue;
            
            BOOLEAN isTracked = FALSE;
            HRESULT hr = body->get_IsTracked(&isTracked);
            
            if (SUCCEEDED(hr) && isTracked) {
                // Create a new tracked body
                KinectInterface::BodyData bodyData;
                bodyData.isTracked = true;
                
                // Get tracking ID
                UINT64 trackingId = 0;
                body->get_TrackingId(&trackingId);
                bodyData.trackingId = static_cast<int>(trackingId);
                
                // Get hand states
                HandState leftHandState = HandState_Unknown;
                HandState rightHandState = HandState_Unknown;
                body->get_HandLeftState(&leftHandState);
                body->get_HandRightState(&rightHandState);
                
                // Store hand states
                handStates[i] = std::make_pair(
                    convertHandState(leftHandState),
                    convertHandState(rightHandState)
                );
                
                // Get joints
                Joint joints[JointType_Count];
                hr = body->GetJoints(_countof(joints), joints);
                
                if (SUCCEEDED(hr)) {
                    // Get depth space points for all joints
                    std::array<DepthSpacePoint, JointType_Count> depthPoints;
                    for (int j = 0; j < _countof(joints); ++j) {
                        coordinateMapper->MapCameraPointToDepthSpace(
                            joints[j].Position, &depthPoints[j]);
                    }
                    
                    // Fill body data
                    for (int j = 0; j < JointType_Count; ++j) {
                        auto& jointData = bodyData.joints[j];
                        const auto& joint = joints[j];
                        const auto& depthPoint = depthPoints[j];
                        
                        jointData.position = cv::Point2f(depthPoint.X, depthPoint.Y);
                        jointData.positionWorld = cv::Point3f(
                            joint.Position.X, joint.Position.Y, joint.Position.Z);
                        jointData.tracked = (joint.TrackingState == TrackingState_Tracked);
                    }
                    
                    // Draw the skeleton
                    drawSkeleton(joints, depthPoints.data(), leftHandState, rightHandState);
                    
                    // Add body data to tracked bodies
                    trackedBodies.push_back(bodyData);
                }
            }
        }
    }

    void drawSkeleton(const Joint* joints, const DepthSpacePoint* depthPoints,
                     HandState leftHandState, HandState rightHandState) {
        // Draw bones
        drawBone(joints, depthPoints, JointType_Head, JointType_Neck);
        drawBone(joints, depthPoints, JointType_Neck, JointType_SpineShoulder);
        drawBone(joints, depthPoints, JointType_SpineShoulder, JointType_SpineMid);
        drawBone(joints, depthPoints, JointType_SpineMid, JointType_SpineBase);
        drawBone(joints, depthPoints, JointType_SpineShoulder, JointType_ShoulderRight);
        drawBone(joints, depthPoints, JointType_SpineShoulder, JointType_ShoulderLeft);
        drawBone(joints, depthPoints, JointType_SpineBase, JointType_HipRight);
        drawBone(joints, depthPoints, JointType_SpineBase, JointType_HipLeft);

        // Right Arm
        drawBone(joints, depthPoints, JointType_ShoulderRight, JointType_ElbowRight);
        drawBone(joints, depthPoints, JointType_ElbowRight, JointType_WristRight);
        drawBone(joints, depthPoints, JointType_WristRight, JointType_HandRight);
        drawBone(joints, depthPoints, JointType_HandRight, JointType_HandTipRight);
        drawBone(joints, depthPoints, JointType_WristRight, JointType_ThumbRight);

        // Left Arm
        drawBone(joints, depthPoints, JointType_ShoulderLeft, JointType_ElbowLeft);
        drawBone(joints, depthPoints, JointType_ElbowLeft, JointType_WristLeft);
        drawBone(joints, depthPoints, JointType_WristLeft, JointType_HandLeft);
        drawBone(joints, depthPoints, JointType_HandLeft, JointType_HandTipLeft);
        drawBone(joints, depthPoints, JointType_WristLeft, JointType_ThumbLeft);

        // Right Leg
        drawBone(joints, depthPoints, JointType_HipRight, JointType_KneeRight);
        drawBone(joints, depthPoints, JointType_KneeRight, JointType_AnkleRight);
        drawBone(joints, depthPoints, JointType_AnkleRight, JointType_FootRight);

        // Left Leg
        drawBone(joints, depthPoints, JointType_HipLeft, JointType_KneeLeft);
        drawBone(joints, depthPoints, JointType_KneeLeft, JointType_AnkleLeft);
        drawBone(joints, depthPoints, JointType_AnkleLeft, JointType_FootLeft);
        
        // Draw hand states
        drawHandState(depthPoints[JointType_HandLeft], leftHandState);
        drawHandState(depthPoints[JointType_HandRight], rightHandState);
    }

    void drawBone(const Joint* joints, const DepthSpacePoint* depthPoints,
                 JointType joint0, JointType joint1) {
        TrackingState joint0State = joints[joint0].TrackingState;
        TrackingState joint1State = joints[joint1].TrackingState;

        // If both joints are not tracked, skip drawing
        if (joint0State == TrackingState_NotTracked || 
            joint1State == TrackingState_NotTracked) {
            return;
        }

        // If both joints are inferred, skip drawing
        if (joint0State == TrackingState_Inferred && 
            joint1State == TrackingState_Inferred) {
            return;
        }

        // Convert joint positions to depth space points
        cv::Point p1(static_cast<int>(depthPoints[joint0].X),
                    static_cast<int>(depthPoints[joint0].Y));
                    
        cv::Point p2(static_cast<int>(depthPoints[joint1].X),
                    static_cast<int>(depthPoints[joint1].Y));

        // Check if points are within image bounds
        if (p1.x >= 0 && p1.x < KinectInterface::Resolution::DepthWidth && 
            p1.y >= 0 && p1.y < KinectInterface::Resolution::DepthHeight &&
            p2.x >= 0 && p2.x < KinectInterface::Resolution::DepthWidth && 
            p2.y >= 0 && p2.y < KinectInterface::Resolution::DepthHeight) {
            
            // Use different colors for tracked vs inferred joints
            cv::Scalar color;
            if (joint0State == TrackingState_Tracked && 
                joint1State == TrackingState_Tracked) {
                // Fully tracked - use yellow
                color = cv::Scalar(255, 255, 0);
            } else {
                // At least one inferred - use blue
                color = cv::Scalar(255, 0, 0);
            }
            
            cv::line(skeletonImage, p1, p2, color, 2);
        }
    }

    void drawHandState(const DepthSpacePoint depthPoint, HandState handState) {
        // Skip if hand state is unknown or not tracked
        if (handState == HandState_Unknown || handState == HandState_NotTracked) {
            return;
        }
        
        // Convert to point
        cv::Point handPoint(static_cast<int>(depthPoint.X), 
                           static_cast<int>(depthPoint.Y));
        
        // Check if point is within image bounds
        if (handPoint.x >= 0 && handPoint.x < KinectInterface::Resolution::DepthWidth && 
            handPoint.y >= 0 && handPoint.y < KinectInterface::Resolution::DepthHeight) {
            
            // Choose color based on hand state
            cv::Scalar color;
            switch (handState) {
                case HandState_Open:
                    color = cv::Scalar(255, 0, 0);  // Blue for open
                    break;
                case HandState_Closed:
                    color = cv::Scalar(0, 255, 0);  // Green for closed
                    break;
                case HandState_Lasso:
                    color = cv::Scalar(0, 0, 255);  // Red for lasso
                    break;
                default:
                    return;  // Skip unknown states
            }
            
            // Draw a circle for the hand state
            cv::circle(skeletonImage, handPoint, 15, color, -1);
        }
    }

    KinectInterface::HandState convertHandState(HandState kinectHandState) {
        switch (kinectHandState) {
            case HandState_Open:
                return KinectInterface::HandState::Open;
            case HandState_Closed:
                return KinectInterface::HandState::Closed;
            case HandState_Lasso:
                return KinectInterface::HandState::Lasso;
            case HandState_NotTracked:
                return KinectInterface::HandState::NotTracked;
            case HandState_Unknown:
            default:
                return KinectInterface::HandState::Unknown;
        }
    }

    cv::Mat visualizeDepth(const cv::Mat& depthMat) const {
        cv::Mat colorized;
        
        // Apply a colormap to the depth image
        if (!depthMat.empty()) {
            // Normalize to 8-bit if needed
            cv::Mat depthNorm;
            if (depthMat.type() == CV_16UC1) {
                cv::normalize(depthMat, depthNorm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            } else {
                depthNorm = depthMat;
            }
            
            // Apply jet colormap
            cv::applyColorMap(depthNorm, colorized, cv::COLORMAP_JET);
        }
        
        return colorized;
    }

    cv::Mat visualizeSkeleton(const cv::Mat& backgroundImage, 
                             const KinectInterface::VisualizationOptions& options) const {
        // Create a copy of the background image
        cv::Mat result;
        if (!backgroundImage.empty()) {
            // Make sure background is 3-channel BGR
            if (backgroundImage.channels() == 4) {
                cv::cvtColor(backgroundImage, result, cv::COLOR_BGRA2BGR);
            } else if (backgroundImage.channels() == 1) {
                cv::cvtColor(backgroundImage, result, cv::COLOR_GRAY2BGR);
            } else {
                result = backgroundImage.clone();
            }
            
            // Overlay skeleton if enabled
            if (options.showSkeleton && !skeletonImage.empty()) {
                // Add alpha blending of the skeleton
                for (int y = 0; y < result.rows && y < skeletonImage.rows; ++y) {
                    for (int x = 0; x < result.cols && x < skeletonImage.cols; ++x) {
                        cv::Vec3b skeleton = skeletonImage.at<cv::Vec3b>(y, x);
                        if (skeleton != cv::Vec3b(0, 0, 0)) {  // Not black (has content)
                            result.at<cv::Vec3b>(y, x) = skeleton;
                        }
                    }
                }
            }

            // Add frame number if enabled
            if (options.addFrameNumber) {
                cv::putText(result, "Frame: " + std::to_string(frameCount),
                           cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                           cv::Scalar(255, 255, 255), 2);
            }
        } else {
            // If no background provided, just return the skeleton
            result = skeletonImage.clone();
        }
        
        return result;
    }

    double measureFrameRate(int durationSeconds) const {
        if (!initialized || !depthFrameReader) {
            spdlog::error("Cannot measure FPS: device not initialized");
            return 0.0;
        }

        int frameCount = 0;
        int failedFrames = 0;
        auto startTime = std::chrono::high_resolution_clock::now();

        spdlog::info("Starting FPS measurement for {} seconds", durationSeconds);

        while (true) {
            IDepthFrame* frame = nullptr;
            HRESULT hr = depthFrameReader->AcquireLatestFrame(&frame);

            if (SUCCEEDED(hr) && frame) {
                frameCount++;
                SafeRelease(frame);
            } else {
                failedFrames++;
            }

            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>
                (currentTime - startTime).count();

            if (elapsedSeconds >= durationSeconds) {
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        double fps = static_cast<double>(frameCount) / durationSeconds;

        spdlog::info("FPS Measurement Results:");
        spdlog::info("  Frames captured: {}", frameCount);
        spdlog::info("  Failed frame reads: {}", failedFrames);
        spdlog::info("  Average FPS: {:.1f}", fps);

        return fps;
    }

    // Member variables
    IKinectSensor* kinectSensor;
    IDepthFrameReader* depthFrameReader;
    IColorFrameReader* colorFrameReader;
    IBodyFrameReader* bodyFrameReader;
    IBodyIndexFrameReader* bodyIndexFrameReader;
    ICoordinateMapper* coordinateMapper;
    
    bool initialized;
    int frameCount;
    
    cv::Mat skeletonImage;
    cv::Mat depthImage;
    cv::Mat colorImage;
    cv::Mat rawDepthImage;
    
    std::vector<KinectInterface::BodyData> trackedBodies;
    std::array<std::pair<KinectInterface::HandState, KinectInterface::HandState>, BODY_COUNT> handStates;
    
    KinectInterface::FrameReadyCallback frameReadyCallback;
};

//
// Main KinectInterface implementation
//

KinectInterface::KinectInterface() 
    : pImpl(std::make_unique<KinectInterfaceImpl>()) {
}

KinectInterface::~KinectInterface() = default;

KinectInterface::KinectInterface(KinectInterface&&) noexcept = default;
KinectInterface& KinectInterface::operator=(KinectInterface&&) noexcept = default;

bool KinectInterface::initialize() {
    return pImpl->initialize();
}

bool KinectInterface::isInitialized() const noexcept {
    return pImpl->initialized;
}

bool KinectInterface::update() {
    return pImpl->update();
}

std::optional<cv::Mat> KinectInterface::getColorFrame() const {
    if (pImpl->colorImage.empty()) {
        return std::nullopt;
    }
    return pImpl->colorImage.clone();
}

std::optional<cv::Mat> KinectInterface::getDepthFrame() const {
    if (pImpl->rawDepthImage.empty()) {
        return std::nullopt;
    }
    return pImpl->rawDepthImage.clone();
}

std::optional<cv::Mat> KinectInterface::getSkeletonVisualization() const {
    if (pImpl->skeletonImage.empty()) {
        return std::nullopt;
    }
    return pImpl->skeletonImage.clone();
}

std::vector<KinectInterface::BodyData> KinectInterface::getTrackedBodies() const {
    return pImpl->trackedBodies;
}

std::optional<KinectInterface::HandState> KinectInterface::getHandState(bool leftHand, int bodyIndex) const {
    if (bodyIndex < 0 || bodyIndex >= BODY_COUNT || pImpl->trackedBodies.empty()) {
        return std::nullopt;
    }
    
    // Find the requested body index within tracked bodies
    if (bodyIndex < static_cast<int>(pImpl->trackedBodies.size())) {
        // Return the hand state for the specified hand
        return leftHand ? 
            pImpl->handStates[bodyIndex].first : 
            pImpl->handStates[bodyIndex].second;
    }
    
    return std::nullopt;
}

ICoordinateMapper* KinectInterface::getCoordinateMapper() const {
    return pImpl->coordinateMapper;
}

double KinectInterface::measureFrameRate(int durationSeconds) const {
    return pImpl->measureFrameRate(durationSeconds);
}

cv::Mat KinectInterface::visualizeDepth(const cv::Mat& depthFrame) const {
    return pImpl->visualizeDepth(depthFrame);
}

cv::Mat KinectInterface::visualizeSkeleton(const cv::Mat& backgroundImage, 
                                         const VisualizationOptions& options) const {
    return pImpl->visualizeSkeleton(backgroundImage, options);
}

void KinectInterface::setFrameReadyCallback(FrameReadyCallback callback) {
    pImpl->frameReadyCallback = std::move(callback);
}