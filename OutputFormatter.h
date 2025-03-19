#ifndef OUTPUT_FORMATTER_H
#define OUTPUT_FORMATTER_H

#include <string>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>

struct JointData {
    float x, y, z;
    float confidence;

    JointData() : x(0), y(0), z(0), confidence(0) {}

    JointData(float x, float y, float z, float confidence)
        : x(x), y(y), z(z), confidence(confidence) {}
};

struct RecordingMetadata {
    std::string recordingId;
    std::string startDateTime;
    int totalFrames = 0;
    double durationSeconds = 0.0;
    float averageFps = 0.0f;
    int depthWidth = 512;
    int depthHeight = 424;
    int rgbWidth = 1920;
    int rgbHeight = 1080;
    std::vector<uint64_t> frameTimestamps;
    std::map<std::string, std::string> additionalInfo;
};

class OutputFormatter {
public:
    struct VisualizationOptions {
        int width = 1280;
        int height = 720;
        bool showJointIds = false;
        bool showDepthValues = true;
        bool showFrameInfo = true;
        std::string frameInfo;
        float minConfidence = 0.2f;
        int jointRadius = 5;
        int lineThickness = 2;
        cv::Scalar highConfidenceColor = cv::Scalar(0, 255, 0);   // Green
        cv::Scalar mediumConfidenceColor = cv::Scalar(0, 255, 255); // Yellow
        cv::Scalar lowConfidenceColor = cv::Scalar(0, 0, 255);    // Red
    };

    OutputFormatter();
    ~OutputFormatter();

    static bool saveToJson(const std::string& filename,
                         const std::vector<std::map<int, JointData>>& framesJointData,
                         const RecordingMetadata& metadata);

    static bool exportToCSV(const std::string& filename,
                          const std::vector<std::map<int, JointData>>& framesJointData,
                          const RecordingMetadata& metadata);

    static bool exportToBinaryFormat(const std::string& filename,
                                   const std::vector<std::map<int, JointData>>& framesJointData,
                                   const RecordingMetadata& metadata);

    static bool saveFrameJson(const std::string& directory,
                            int frameIndex,
                            const std::map<int, JointData>& joints,
                            uint64_t timestamp = 0);

    static cv::Mat createJointVisualization(const std::map<int, JointData>& joints,
                                          const cv::Mat& backgroundImage = cv::Mat(),
                                          const VisualizationOptions& options = {});
};

#endif