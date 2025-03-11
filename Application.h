#pragma once

#include "Configuration.h"
#include "KinectInterface.h"
#include "FrameRecorder.h"
#include "OpenPoseInterface.h"

#include <string>
#include <memory>
#include <filesystem>
#include <optional>
#include <functional>
#include <future>

/**
 * @brief Main application class for KinectV2 OpenPose Integration
 * 
 * This class provides a high-level interface for recording and processing
 * Kinect data with OpenPose, handling all the underlying details.
 */
class Application {
public:
    // Application operation modes
    enum class Mode {
        Record,
        Process
    };
    
    // UI callback for rendering
    using RenderCallback = std::function<void(const cv::Mat& frame)>;
    
    // Progress callback for processing
    using ProgressCallback = std::function<void(int current, int total, const std::string& status)>;
    
    // Constructor
    explicit Application(const std::string& configPath = "config.ini");
    
    // No copy
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
    
    // Initialize the application
    bool initialize(Mode mode = Mode::Record);
    
    // Run the application
    void run();
    
    // Stop the application
    void stop();
    
    // Set render callback for UI updates
    void setRenderCallback(RenderCallback callback);
    
    // Set progress callback for processing
    void setProgressCallback(ProgressCallback callback);
    
    // Recording control
    bool startRecording(const std::string& sessionId = "");
    bool stopRecording();
    bool isRecording() const noexcept;

    // Set the session ID for recording
    void setSessionId(const std::string& id);
    
    // Processing control
    bool startProcessing(const std::string& recordingPath, const std::string& outputPath = "");
    bool isProcessing() const noexcept;
    std::optional<OpenPoseInterface::ProcessingResult> getProcessingResult() const;
    
    // Configuration access
    Configuration& getConfig() noexcept;
    void setRecordingOptions(const FrameRecorder::RecordingOptions& options);
    const FrameRecorder::RecordingOptions& getRecordingOptions() const;
    void setOpenPoseConfig(const OpenPoseInterface::Configuration& config) const;
    const OpenPoseInterface::Configuration& getOpenPoseConfig() const;
    
    // Handle keyboard input
    void handleKeypress(int key);

private:
    // Application components
    std::unique_ptr<Configuration> config;
    std::unique_ptr<KinectInterface> kinect;
    std::unique_ptr<FrameRecorder> recorder;
    std::unique_ptr<OpenPoseInterface> openpose;
    
    // Application state
    Mode currentMode;
    bool initialized{false};
    bool running{false};
    std::string sessionId;
    
    // Rendering
    RenderCallback renderCallback;
    ProgressCallback progressCallback;
    
    // Processing state
    std::future<OpenPoseInterface::ProcessingResult> processingFuture;
    std::optional<OpenPoseInterface::ProcessingResult> processingResult;
    std::atomic<bool> processingActive{false};
    
    // Helper methods
    bool initializeRecordMode();
    bool initializeProcessMode();
    void updateFrame();
    void renderFrame(const cv::Mat& frame);
    void updateProcessingProgress(int current, int total);
    static std::string generateSessionId();
};