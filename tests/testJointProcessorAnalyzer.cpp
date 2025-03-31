#include "../JointProcessorAnalyzer.h"
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) {
        spdlog::error("Failed to initialize COM");
        return -1;
    }

    try {
        // Create the analyzer
        JointProcessorAnalyzer analyzer;
        
        // Specify recording path
        std::string recordingName = "testUdlap0120250312_090509";
        if (argc > 1) {
            recordingName = argv[1];
        }
        
        // Try to find the recording in both processed and recordings directories
        fs::path basePath = fs::current_path();
        fs::path processedPath = basePath / "processed" / recordingName;
        fs::path recordingsPath = basePath / "recordings" / recordingName;

        // Load the recording
        spdlog::info("Loading recording data...");
        if (!analyzer.loadRecording(recordingsPath, processedPath)) {
            spdlog::error("Failed to load recording");
            CoUninitialize();
            return -1;
        }
        
        // Process the data 
        spdlog::info("Processing joint data...");
        if (!analyzer.processData()) {
            spdlog::error("Failed to process data");
            CoUninitialize();
            return -1;
        }
        
        // Analyze and visualize results
        int frameCount = analyzer.getFrameCount();
        spdlog::info("Successfully processed {} frames", frameCount);
        
        // Create output directory for visualizations
        fs::path outputDir = processedPath / "analysis";
        fs::create_directories(outputDir);
        
        // Generate side-by-side comparison for some frames
        spdlog::info("Generating visualizations...");
        
        // Sample frames throughout the recording
        std::vector<int> sampleFrames;
        if (frameCount <= 10) {
            // If few frames, use all of them
            for (int i = 0; i < frameCount; i++) {
                sampleFrames.push_back(i);
            }
        } else {
            // Sample 10 frames evenly distributed
            for (int i = 0; i < 10; i++) {
                int frameIdx = (i * frameCount) / 10;
                sampleFrames.push_back(frameIdx);
            }
        }
        
        // Generate side-by-side comparisons
        for (int frameIdx : sampleFrames) {
            cv::Mat sideBySide = analyzer.visualizeSideBySide(frameIdx);
            fs::path outputPath = outputDir / ("frame_" + std::to_string(frameIdx) + "_comparison.png");
            cv::imwrite(outputPath.string(), sideBySide);
            spdlog::info("Saved comparison for frame {}", frameIdx);
        }
        
        // Generate motion trails visualization
        int midFrame = frameCount / 2;
        cv::Mat motionTrails = analyzer.visualizeMotionTrails(midFrame, 30);
        cv::imwrite((outputDir / "motion_trails.png").string(), motionTrails);
        spdlog::info("Saved motion trails visualization");
        
        // Generate joint angle constraints visualization
        cv::Mat angleConstraints = analyzer.visualizeJointAngleConstraints(midFrame);
        cv::imwrite((outputDir / "angle_constraints.png").string(), angleConstraints);
        spdlog::info("Saved joint angle constraints visualization");
        
        // Generate stability metrics for the whole recording
        cv::Mat stabilityMetrics = analyzer.visualizeStabilityMetrics(0, frameCount-1, 1280, 720);
        cv::imwrite((outputDir / "stability_metrics.png").string(), stabilityMetrics);
        spdlog::info("Saved stability metrics visualization");
        
        // Export analysis data to JSON
        analyzer.exportToJson(outputDir / "analysis_data.json");
        spdlog::info("Exported analysis data to JSON");
        
        // Generate a full analysis video
        spdlog::info("Generating analysis video (this may take a while)...");
        analyzer.generateAnalysisVideo(outputDir / "analysis_video.mp4", true, true, true);
        spdlog::info("Analysis video generated successfully");
        
        spdlog::info("Analysis complete! Results saved to: {}", outputDir.string());
        
        // Cleanup
        CoUninitialize();
        return 0;
    } catch (const std::exception& e) {
        spdlog::error("Unhandled exception: {}", e.what());
        CoUninitialize();
        return -1;
    }
}