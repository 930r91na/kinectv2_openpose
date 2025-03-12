#include "Application.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <iomanip>
#include <thread>
#include <opencv2/opencv.hpp>

Application::Application(const std::string& configPath)
    : config(std::make_unique<Configuration>(configPath)),
      currentMode(Mode::Record) {

    spdlog::info("Application created");
}

bool Application::initialize(Mode mode) {
    if (initialized) {
        spdlog::warn("Application already initialized");
        return true;
    }

    currentMode = mode;

    // Initialize components based on mode
    if (mode == Mode::Record) {
        return initializeRecordMode();
    } else {
        return initializeProcessMode();
    }
}

bool Application::initializeRecordMode() {
    // Initialize Kinect
    kinect = std::make_unique<KinectInterface>();

    if (!kinect->initialize()) {
        spdlog::error("Failed to initialize Kinect");
        return false;
    }

    // Initialize recorder with options from config
    FrameRecorder::RecordingOptions recorderOptions;
    recorderOptions.useVideoCompression = config->getValueOr<bool>("use_video_compression", true);
    recorderOptions.targetFps = config->getValueOr<int>("frame_rate_limit", 30);
    recorderOptions.limitFrameRate = config->getValueOr<bool>("enable_frame_limiting", true);
    recorderOptions.processEveryNFrames = config->getValueOr<int>("process_every_n_frames", 15);
    recorderOptions.outputDirectory = config->getValueOr<std::string>("recording_directory", "recordings");

    recorder = std::make_unique<FrameRecorder>(recorderOptions);

    // We don't need OpenPose in record mode
    openpose.reset();

    initialized = true;
    spdlog::info("Application initialized in record mode");
    return true;
}

bool Application::initializeProcessMode() {
    // Initialize Kinect (needed for coordinate mapper)
    kinect = std::make_unique<KinectInterface>();

    if (!kinect->initialize()) {
        spdlog::error("Failed to initialize Kinect");
        return false;
    }

    // Initialize OpenPose with options from config
    OpenPoseInterface::Configuration opConfig;
    opConfig.openPoseExePath = config->getPathOr("openpose_path", "bin/OpenPoseDemo.exe");
    opConfig.netResolution = config->getValueOr<int>("net_resolution", 368);
    opConfig.useMaximumAccuracy = config->getValueOr<bool>("use_maximum_accuracy", false);
    opConfig.keypointConfidenceThreshold = config->getValueOr<int>("keypoint_confidence_threshold", 40);
    opConfig.performanceMode = config->getValueOr<bool>("performance_mode", false);

    openpose = std::make_unique<OpenPoseInterface>(opConfig);

    if (!openpose->initialize()) {
        spdlog::error("Failed to initialize OpenPose");
        return false;
    }

    // We don't need recorder in process mode
    recorder.reset();

    initialized = true;
    spdlog::info("Application initialized in process mode");
    return true;
}

void Application::run() {
    if (!initialized) {
        spdlog::error("Cannot run: application not initialized");
        return;
    }

    running = true;

    // Start the main loop based on mode
    if (currentMode == Mode::Record) {
        spdlog::info("Starting recording mode");

        // Create window for display if not using external rendering
        if (!renderCallback) {
            cv::namedWindow("Kinect Recorder", cv::WINDOW_NORMAL);
            cv::resizeWindow("Kinect Recorder", 1280, 720);
        }

        // Main recording loop
        while (running) {
            // Update frame
            updateFrame();

            // Process keyboard input
            int key = cv::waitKey(1);
            if (key >= 0) {
                handleKeypress(key);
            }

            // Sleep to avoid maxing out CPU
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Clean up
        if (!renderCallback) {
            cv::destroyAllWindows();
        }
    } else {
        spdlog::info("Starting process mode");

        // Process mode is generally non-interactive
        // Just wait for processing to complete
        if (processingActive && processingFuture.valid()) {
            if (progressCallback) {
                progressCallback(0, 100, "Waiting for processing to complete...");
            }

            processingResult = processingFuture.get();
            processingActive = false;

            if (progressCallback) {
                progressCallback(100, 100, "Processing complete");
            }

            spdlog::info("Processing complete");
        } else {
            spdlog::warn("No processing task is active");
        }

        running = false;
    }

    spdlog::info("Application stopped");
}

void Application::stop() {
    running = false;

    // Stop recording if active
    if (isRecording()) {
        stopRecording();
    }

    spdlog::info("Application stopping...");
}

void Application::setRenderCallback(RenderCallback callback) {
    renderCallback = std::move(callback);
}

void Application::setProgressCallback(ProgressCallback callback) {
    progressCallback = std::move(callback);
}

bool Application::startRecording(const std::string& session) {
    if (currentMode != Mode::Record || !recorder) {
        spdlog::error("Cannot start recording: not in record mode");
        return false;
    }

    // If a session ID is provided, use it; otherwise use the already set sessionId
    std::string actualSessionId;

    if (!session.empty()) {
        actualSessionId = session;
    } else if (!sessionId.empty()) {
        actualSessionId = sessionId;
    } else {
        actualSessionId = generateSessionId();
    }

    // Update the current session ID
    sessionId = actualSessionId;

    spdlog::info("Starting recording session: {}", sessionId);

    // Pass the session ID directly to the recorder
    return recorder->startRecording(sessionId);
}

bool Application::stopRecording() {
    if (!recorder || !recorder->isRecording()) {
        return false;
    }

    spdlog::info("Stopping recording");
    return recorder->stopRecording();
}

bool Application::isRecording() const noexcept {
    return recorder && recorder->isRecording();
}

void Application::setSessionId(const std::string& id)
{
    sessionId = id;
    spdlog::info("Session ID set to: {}", sessionId);
}

bool Application::startProcessing(const std::string& recordingPath, const std::string& outputPath) {
    if (currentMode != Mode::Process || !openpose) {
        spdlog::error("Cannot start processing: not in process mode");
        return false;
    }

    if (processingActive) {
        spdlog::warn("Processing already in progress");
        return false;
    }

    // Determine output path
    std::string actualOutputPath;
    if (outputPath.empty()) {
        actualOutputPath = config->getValueOr<std::string>("output_directory", "processed") +
            "/" + std::filesystem::path(recordingPath).filename().string();
    } else {
        actualOutputPath = outputPath;
    }

    spdlog::info("Starting processing of: {}", recordingPath);
    spdlog::info("Output will be saved to: {}", actualOutputPath);

    // Start processing asynchronously
    processingActive = true;
    processingResult = std::nullopt;

    processingFuture = openpose->processRecordingAsync(
        recordingPath,
        kinect->getCoordinateMapper(),
        actualOutputPath,
        config->getValueOr<int>("processing_threads", 4),
        [this](int current, int total) {
            updateProcessingProgress(current, total);
        }
    );

    return true;
}

bool Application::isProcessing() const noexcept {
    return processingActive;
}

std::optional<OpenPoseInterface::ProcessingResult> Application::getProcessingResult() const {
    return processingResult;
}

Configuration& Application::getConfig()  noexcept {
    return *config;
}

void Application::setRecordingOptions(const FrameRecorder::RecordingOptions& options) {
    if (recorder) {
        recorder->setRecordingOptions(options);

        // Update config to persist changes
        config->setValue("use_video_compression", options.useVideoCompression);
        config->setValue("frame_rate_limit", options.targetFps);
        config->setValue("enable_frame_limiting", options.limitFrameRate);
        config->setValue("process_every_n_frames", options.processEveryNFrames);
        config->setValue("recording_directory", options.outputDirectory);

        config->saveToFile(config->getConfigFilePath());
    }
}

const FrameRecorder::RecordingOptions& Application::getRecordingOptions() const {
    if (recorder) {
        return recorder->getRecordingOptions();
    }

    static FrameRecorder::RecordingOptions defaultOptions;
    return defaultOptions;
}

void Application::setOpenPoseConfig(const OpenPoseInterface::Configuration& opConfig) const
{
    if (openpose) {
        openpose->setConfiguration(opConfig);

        // Update config to persist changes
        config->setValue("openpose_path", opConfig.openPoseExePath.string());
        config->setValue("net_resolution", opConfig.netResolution);
        config->setValue("use_maximum_accuracy", opConfig.useMaximumAccuracy);
        config->setValue("keypoint_confidence_threshold", opConfig.keypointConfidenceThreshold);
        config->setValue("performance_mode", opConfig.performanceMode);

        config->saveToFile(config->getConfigFilePath());
    }
}

const OpenPoseInterface::Configuration& Application::getOpenPoseConfig() const {
    if (openpose) {
        return openpose->getConfiguration();
    }

    static OpenPoseInterface::Configuration defaultConfig;
    return defaultConfig;
}

void Application::handleKeypress(int key) {
    // Common keys
    if (key == 27 || key == 'q' || key == 'Q') { // ESC or Q
        stop();
        return;
    }

    // Mode-specific keys
    if (currentMode == Mode::Record) {
        if (key == 'r' || key == 'R') { // Start/stop recording
            if (isRecording()) {
                stopRecording();
            } else {
                startRecording(sessionId); // Use the current sessionId
            }
        } else if (key == 's' || key == 'S') { // Save recording
            if (isRecording()) {
                stopRecording();
            }
        }
    }
}

void Application::updateFrame() {
    if (!kinect || !initialized) {
        return;
    }

    // Update Kinect frame
    if (!kinect->update()) {
        // Skip rendering if no new frame
        return;
    }

    // Get the color frame
    auto colorFrame = kinect->getColorFrame();
    if (!colorFrame) {
        return;
    }

    // Get depth frame
    auto depthFrame = kinect->getDepthFrame();

    // Record frame if recording is active
    if (isRecording() && depthFrame && colorFrame) {
        recorder->addFrame(*colorFrame, *depthFrame);
    }

    // Create visualization
    cv::Mat displayFrame;
    if (colorFrame) {
        // Create a blank canvas with dark background
        displayFrame = cv::Mat(720, 1280, CV_8UC3, cv::Scalar(40, 40, 40));

        // Skeleton window visualizations
        int maxSkeletonWidth = 600;
        int maxSkeletonHeight = 400;

        // Resize color frame for display with proper dimensions
        cv::Mat resizedColor;
        double scaleX = static_cast<double>(maxSkeletonWidth) / colorFrame->cols;
        double scaleY = static_cast<double>(maxSkeletonHeight) / colorFrame->rows;
        double scale = std::min(scaleX, scaleY);
        cv::resize(*colorFrame, resizedColor, cv::Size(), scale, scale);

        // Get skeleton overlay
        cv::Mat skeletonViz = kinect->visualizeSkeleton(resizedColor);

        // Create depth visualization
        cv::Mat depthViz;
        if (depthFrame) {
            depthViz = kinect->visualizeDepth(*depthFrame);

            // Use safe resize to ensure it fits
            int depthWidth = std::min(320, displayFrame.cols / 4);
            int depthHeight = std::min(240, displayFrame.rows / 4);
            cv::resize(depthViz, depthViz, cv::Size(depthWidth, depthHeight));
        } else {
            // Create a properly sized empty depth visualization
            int depthWidth = std::min(320, displayFrame.cols / 4);
            int depthHeight = std::min(240, displayFrame.rows / 4);
            depthViz = cv::Mat(depthHeight, depthWidth, CV_8UC3, cv::Scalar(0, 0, 0));
        }

        // Calculate layout - left panel for main visualization, right panel for controls
        int leftPanelWidth = 640; // Half of 1280
        int rightPanelStartX = leftPanelWidth + 20; // Add some margin
        int panelMargin = 20;

        // Left panel - main skeleton visualization
        // Make sure we don't exceed the display frame dimensions
        int skeletonWidth = std::min(skeletonViz.cols, leftPanelWidth - 2*panelMargin);
        int skeletonHeight = std::min(skeletonViz.rows, displayFrame.rows - 2*panelMargin - 240 - panelMargin);
        if (skeletonWidth > 0 && skeletonHeight > 0) {
            // Resize to fit if needed
            cv::Mat resizedSkeletonViz = skeletonViz;
            if (skeletonViz.cols != skeletonWidth || skeletonViz.rows != skeletonHeight) {
                cv::resize(skeletonViz, resizedSkeletonViz, cv::Size(skeletonWidth, skeletonHeight));
            }

            // Create ROI and copy
            cv::Rect skeletonRoi(panelMargin, panelMargin, resizedSkeletonViz.cols, resizedSkeletonViz.rows);
            if (skeletonRoi.x >= 0 && skeletonRoi.y >= 0 &&
                skeletonRoi.x + skeletonRoi.width <= displayFrame.cols &&
                skeletonRoi.y + skeletonRoi.height <= displayFrame.rows) {
                resizedSkeletonViz.copyTo(displayFrame(skeletonRoi));
            }
        }

        // Left panel - depth visualization below skeleton
        if (!depthViz.empty()) {
            int depthY = 2*panelMargin + skeletonHeight;
            // Make sure the depth visualization fits
            int depthWidth = std::min(depthViz.cols, leftPanelWidth - 2*panelMargin);
            int depthHeight = std::min(depthViz.rows, displayFrame.rows - depthY - panelMargin);

            if (depthWidth > 0 && depthHeight > 0) {
                // Resize depth visualization if needed
                cv::Mat resizedDepthViz = depthViz;
                if (depthViz.cols != depthWidth || depthViz.rows != depthHeight) {
                    cv::resize(depthViz, resizedDepthViz, cv::Size(depthWidth, depthHeight));
                }

                // Create ROI and copy
                cv::Rect depthRoi(panelMargin, depthY, resizedDepthViz.cols, resizedDepthViz.rows);
                if (depthRoi.x >= 0 && depthRoi.y >= 0 &&
                    depthRoi.x + depthRoi.width <= displayFrame.cols &&
                    depthRoi.y + depthRoi.height <= displayFrame.rows) {
                    resizedDepthViz.copyTo(displayFrame(depthRoi));
                }
            }
        }

        // Draw a separator line between panels if it fits in the frame
        if (leftPanelWidth > 0 && leftPanelWidth < displayFrame.cols) {
            cv::line(displayFrame,
                    cv::Point(leftPanelWidth, 0),
                    cv::Point(leftPanelWidth, displayFrame.rows),
                    cv::Scalar(100, 100, 100), 2);
        }

        // Make sure we don't exceed the display boundaries
        // Calculate safe right panel dimensions
        int rightPanelWidth = displayFrame.cols - rightPanelStartX - panelMargin;
        if (rightPanelWidth < 100) {
            // Adjust if there's not enough space
            rightPanelStartX = displayFrame.cols - 400;
            rightPanelWidth = 380;
        }

        // Safe text positioning helper function (to avoid drawing outside frame)
        auto safeText = [&displayFrame](const std::string& text, cv::Point position,
                                        double fontScale, cv::Scalar color, int thickness) {
            if (position.x >= 0 && position.y >= 0 &&
                position.x < displayFrame.cols && position.y < displayFrame.rows) {
                cv::putText(displayFrame, text, position, cv::FONT_HERSHEY_SIMPLEX,
                           fontScale, color, thickness);
            }
        };

        // Right panel - Status section
        int statusY = 60;
        safeText("STATUS", cv::Point(rightPanelStartX, statusY), 0.9, cv::Scalar(200, 200, 200), 2);

        // Recording status indicator
        std::string statusText = isRecording() ? "RECORDING" : "READY";
        safeText(statusText, cv::Point(rightPanelStartX + 150, statusY), 0.9,
                isRecording() ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 2);

        // Add recording indicator dot
        if (isRecording()) {
            cv::Point circleCenter(rightPanelStartX + 130, statusY - 5);
            if (circleCenter.x >= 0 && circleCenter.y >= 0 &&
                circleCenter.x < displayFrame.cols && circleCenter.y < displayFrame.rows) {
                cv::circle(displayFrame, circleCenter, 10, cv::Scalar(0, 0, 255), -1);
            }
        }

        // Right panel - Session info section
        int sessionY = std::min(statusY + 80, displayFrame.rows - 200);
        safeText("SESSION INFO", cv::Point(rightPanelStartX, sessionY),
                0.9, cv::Scalar(200, 200, 200), 2);

        // Session ID
        safeText("ID: " + sessionId, cv::Point(rightPanelStartX, sessionY + 40),
                0.7, cv::Scalar(200, 200, 200), 1);

        // Frame count if recording
        if (isRecording() && recorder) {
            safeText("Frames: " + std::to_string(recorder->getFrameCount()),
                    cv::Point(rightPanelStartX, sessionY + 80),
                    0.7, cv::Scalar(200, 200, 200), 1);
        }

        // Right panel - Instructions section (ensure it fits in the display)
        int instructionsY = std::min(sessionY + 140, displayFrame.rows - 120);
        safeText("CONTROLS", cv::Point(rightPanelStartX, instructionsY),
                0.9, cv::Scalar(200, 200, 200), 2);

        // Only add control instructions if there's enough space
        if (instructionsY + 120 < displayFrame.rows) {
            safeText("Press 'r' - Start/Stop recording",
                    cv::Point(rightPanelStartX, instructionsY + 40),
                    0.7, cv::Scalar(200, 200, 200), 1);

            safeText("Press 's' - Save the recording",
                    cv::Point(rightPanelStartX, instructionsY + 80),
                    0.7, cv::Scalar(200, 200, 200), 1);

            safeText("Press 'q' - Quit",
                    cv::Point(rightPanelStartX, instructionsY + 120),
                    0.7, cv::Scalar(200, 200, 200), 1);
        }
    }

    // Render the frame
    if (!displayFrame.empty()) {
        renderFrame(displayFrame);
    }
}

void Application::renderFrame(const cv::Mat& frame) {
    if (renderCallback) {
        // Use custom rendering callback
        renderCallback(frame);
    } else {
        // Use OpenCV rendering
        cv::imshow("Kinect Recorder", frame);
    }
}

void Application::updateProcessingProgress(int current, int total) {
    // Calculate percentage
    int percentage = (total > 0) ? (current * 100 / total) : 0;

    // Update progress via callback
    if (progressCallback) {
        progressCallback(current, total, "Processing frame " + std::to_string(current) +
                        " of " + std::to_string(total));
    }

    // Log progress periodically
    if (current % 10 == 0 || current == total) {
        spdlog::info("Processing progress: {}% ({}/{})", percentage, current, total);
    }
}

std::string Application::generateSessionId()
{
    // Create a timestamp-based session ID
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;

    ss << "recording_" << std::put_time(std::localtime(&timestamp), "%Y%m%d_%H%M%S");

    return ss.str();
}