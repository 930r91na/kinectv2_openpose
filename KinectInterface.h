#pragma once

#include <Kinect.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <functional>
#include <optional>
#include <array>

// Forward declaration for the implementation
class KinectInterfaceImpl;

/**
 * @brief Modern RAII wrapper for the Kinect v2 sensor
 *
 * This class provides a clean, modern C++ interface to the Kinect v2 sensor,
 * handling all resource management and providing access to color, depth, and
 * body tracking data.
 */
class KinectInterface {
public:
    // Frame resolution constants
    struct Resolution {
        static constexpr int DepthWidth = 512;
        static constexpr int DepthHeight = 424;
        static constexpr int ColorWidth = 1920;
        static constexpr int ColorHeight = 1080;
    };

    // Joint data for external use
    struct JointData {
        cv::Point2f position;      // 2D position in depth space
        cv::Point3f positionWorld; // 3D position in world space
        bool tracked;              // Whether the joint is tracked
    };

    // Body data for external use
    struct BodyData {
        std::array<JointData, JointType_Count> joints;
        int trackingId;
        bool isTracked;
    };

    // Hand state for external use
    enum class HandState {
        Unknown,
        NotTracked,
        Open,
        Closed,
        Lasso
    };

    // Visualization options
    struct VisualizationOptions {
        bool showSkeleton = true;
        bool showHandState = true;
        bool showDepth = true;
        bool addFrameNumber = true;
        cv::Scalar trackingColor = cv::Scalar(255, 255, 0);
        cv::Scalar inferredColor = cv::Scalar(255, 0, 0);
    };

    // Constructor and destructor
    KinectInterface();
    ~KinectInterface();

    // Disable copying
    KinectInterface(const KinectInterface&) = delete;
    KinectInterface& operator=(const KinectInterface&) = delete;

    // Enable moving
    KinectInterface(KinectInterface&&) noexcept;
    KinectInterface& operator=(KinectInterface&&) noexcept;

    // Initialization
    bool initialize();
    bool isInitialized() const noexcept;

    // Frame update
    bool update();

    // Frame access methods (using std::optional to indicate potential absence)
    std::optional<cv::Mat> getColorFrame() const;
    std::optional<cv::Mat> getDepthFrame() const;
    std::optional<cv::Mat> getBodyIndexFrame() const;
    std::optional<cv::Mat> getSkeletonVisualization() const;

    // Get tracked bodies
    std::vector<BodyData> getTrackedBodies() const;

    // Get hand states
    std::optional<HandState> getHandState(bool leftHand, int bodyIndex = 0) const;

    // Access to the Kinect's coordinate mapper (needed for 3D reconstruction)
    ICoordinateMapper* getCoordinateMapper() const;

    // Performance measurement
    double measureFrameRate(int durationSeconds = 5) const;

    // Visualization
    cv::Mat visualizeDepth(const cv::Mat& depthFrame) const;
    cv::Mat visualizeSkeleton(const cv::Mat& backgroundImage,
                             const VisualizationOptions& options = {}) const;

    // Event callbacks
    using FrameReadyCallback = std::function<void()>;
    void setFrameReadyCallback(FrameReadyCallback callback);

private:
    // Implementation details hidden with PIMPL idiom
    std::unique_ptr<KinectInterfaceImpl> pImpl;
};