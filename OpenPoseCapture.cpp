#include "OpenPoseCapture.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <chrono>
#include <utility>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

OpenPoseCapture::OpenPoseCapture(std::string openPosePath,
                               std::string tempDir,
                               std::string jsonDir)
    : openPoseExePath(std::move(openPosePath)),
      tempImageDir(std::move(tempDir)),
      outputJsonDir(std::move(jsonDir)),
      netResolution(368),
      useMaximumAccuracy(false),
      keypointConfidenceThreshold(30),
      batchSize(20),
      m_pCoordinateMapper(nullptr) {

    // Create required directories if they don't exist
    if (!fs::exists(tempImageDir)) {
        fs::create_directories(tempImageDir);
    }

    if (!fs::exists(outputJsonDir)) {
        fs::create_directories(outputJsonDir);
    }

    // Initialize joint mapping between OpenPose and Kinect
    initJointMapping();
}

OpenPoseCapture::~OpenPoseCapture() {
    // Stop any processing
    if (isProcessing) {
        isProcessing = false;
        jobCV.notify_all();

        for (auto& thread : workerThreads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    // Clean up temp directories
    try {
        if (fs::exists(tempImageDir)) {
            fs::remove_all(tempImageDir);
        }

        if (fs::exists(outputJsonDir)) {
            fs::remove_all(outputJsonDir);
        }
    } catch (const std::exception& e) {
        spdlog::warn("Failed to clean up temp directory: {}", e.what());
    }
}

void OpenPoseCapture::initJointMapping() {
    // Map OpenPose BODY_25 keypoints to Kinect joint types
    // This mapping may need adjustment based on your specific needs
    openposeToKinectJointMap = {
        {0, JointType_Head},            // Nose -> Head
        {1, JointType_Neck},            // Neck -> Neck
        {2, JointType_ShoulderRight},   // RShoulder -> ShoulderRight
        {3, JointType_ElbowRight},      // RElbow -> ElbowRight
        {4, JointType_WristRight},      // RWrist -> WristRight
        {5, JointType_ShoulderLeft},    // LShoulder -> ShoulderLeft
        {6, JointType_ElbowLeft},       // LElbow -> ElbowLeft
        {7, JointType_WristLeft},       // LWrist -> WristLeft
        {8, JointType_SpineBase},       // MidHip -> SpineBase
        {9, JointType_HipRight},        // RHip -> HipRight
        {10, JointType_KneeRight},      // RKnee -> KneeRight
        {11, JointType_AnkleRight},     // RAnkle -> AnkleRight
        {12, JointType_HipLeft},        // LHip -> HipLeft
        {13, JointType_KneeLeft},       // LKnee -> KneeLeft
        {14, JointType_AnkleLeft},      // LAnkle -> AnkleLeft
        {15, JointType_Head},       // REye -> EyeRight (approximate)
        {16, JointType_Head},        // LEye -> EyeLeft (approximate)
        {17, JointType_Head},           // REar -> Head (approximate)
        {18, JointType_Head},           // LEar -> Head (approximate)
        // Body_25 has additional points (19-24) for feet that don't map well to Kinect
    };
}

bool OpenPoseCapture::initialize() {
    // Check if OpenPose executable exists
    if (!fs::exists(openPoseExePath)) {
        spdlog::error("OpenPose executable not found at: {}", openPoseExePath);
        return false;
    }

    spdlog::info("OpenPoseCapture initialized with executable: {}", openPoseExePath);
    return true;
}

bool OpenPoseCapture::runOpenPoseOnImage(const std::string& imagePath, const std::string& outputDir) const {
    // Extract the OpenPose root directory from the executable path
    std::string openPoseDir = openPoseExePath;
    size_t binPos = openPoseDir.find("\\bin\\");
    if (binPos == std::string::npos) {
        spdlog::error("Cannot determine OpenPose root directory from path: {}", openPoseExePath);
        return false;
    }
    openPoseDir = openPoseDir.substr(0, binPos);

    // Get absolute paths for input and output directories
    std::string absImagePath = fs::absolute(imagePath).string();
    std::string absOutputDir = fs::absolute(outputDir).string();

    // Build command that first changes to OpenPose directory then runs the executable
    std::stringstream cmd;
    cmd << "cd /d \"" << openPoseDir << "\" && "
        << "bin\\OpenPoseDemo.exe"  // Use relative path since we're already in the OpenPose directory
        << " --image_dir \"" << absImagePath << "\""
        << " --write_json \"" << absOutputDir << "\""
        << " --display 0"
        << " --render_pose 0"
        << " --number_people_max 1" // Limit to one person for faster processing
        << " --disable_blending true"; // Disable blending for better performance

    // Performance settings
    if (performanceMode) {
        // EXTREME OPTIMIZATION: These settings sacrifice some accuracy for much faster processing
        cmd << " --net_resolution -1x" << 256; // Lower resolution
        cmd << " --model_pose BODY_25";  // Use COCO model (much faster than BODY_25)
        cmd << " --maximize_positives false";    // Skip low confidence detections
    } else if (useMaximumAccuracy) {
        // Use high accuracy settings
        cmd << " --net_resolution 1312x736"
            << " --scale_number 4"
            << " --scale_gap 0.25";
    } else {
        // Default balanced settings
        cmd << " --net_resolution -1x" << netResolution;
    }

    spdlog::debug("Running OpenPose command: {}", cmd.str());

    // Execute command
    int result = std::system(cmd.str().c_str());

    if (result != 0) {
        spdlog::error("OpenPose execution failed with code: {}", result);
        return false;
    }

    return true;
}

json OpenPoseCapture::readOpenPoseJson(const std::string& jsonPath) {
    std::ifstream file(jsonPath);
    if (!file.is_open()) {
        spdlog::error("Failed to open JSON file: {}", jsonPath);
        return {};
    }

    try {
        json j;
        file >> j;
        return j;
    } catch (const std::exception& e) {
        spdlog::error("Error parsing JSON: {}", e.what());
        return {};
    }
}

std::string OpenPoseCapture::getLastJsonFile(const std::string& directory) {
    std::string latestFile;
    std::time_t latestTime = 0;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() == ".json") {
            auto fileTime = fs::last_write_time(entry.path());
            auto timePoint = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                fileTime - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
            auto time = std::chrono::system_clock::to_time_t(timePoint);

            if (time > latestTime) {
                latestTime = time;
                latestFile = entry.path().string();
            }
        }
    }

    return latestFile;
}

cv::Mat OpenPoseCapture::loadRawDepthData(const std::string& depthPath) {
    std::ifstream depthFile(depthPath, std::ios::binary);
    if (!depthFile.is_open()) {
        spdlog::error("Failed to open depth file: {}", depthPath);
        return cv::Mat();
    }

    // Read dimensions
    int rows, cols;
    depthFile.read(reinterpret_cast<char*>(&rows), sizeof(int));
    depthFile.read(reinterpret_cast<char*>(&cols), sizeof(int));

    // Allocate matrix
    cv::Mat depthImg(rows, cols, CV_16UC1);

    // Read data
    depthFile.read(reinterpret_cast<char*>(depthImg.data),
                  depthImg.total() * depthImg.elemSize());
    depthFile.close();

    return depthImg;
}

bool OpenPoseCapture::process3DLifting(const cv::Mat& colorImage,
                                     const cv::Mat& depthImage,
                                     ICoordinateMapper* coordinateMapper,
                                     const json& openposeData,
                                     std::vector<Person3D>& detectedPeople) const {
    if (colorImage.empty() || depthImage.empty() || !coordinateMapper || openposeData.empty()) {
        spdlog::error("Invalid inputs to process3DLifting");
        return false;
    }

    // Ensure depth image is 16-bit
    cv::Mat depthMat16U;
    if (depthImage.type() != CV_16UC1) {
        depthImage.convertTo(depthMat16U, CV_16UC1, 65535.0/255.0);
    } else {
        depthMat16U = depthImage;
    }

    // Pre-map the entire color frame to depth space for efficiency
    const int depthWidth = depthImage.cols;
    const int depthHeight = depthImage.rows;
    const int depthSize = depthWidth * depthHeight;

    // Create depth to camera mapping table
    std::vector<DepthSpacePoint> colorToDepthMap(colorImage.cols * colorImage.rows);

    // Get depth frame data
    std::vector<UINT16> depthData(depthSize);
    for (int y = 0; y < depthHeight; y++) {
        for (int x = 0; x < depthWidth; x++) {
            depthData[y * depthWidth + x] = depthMat16U.at<UINT16>(y, x);
        }
    }

    // Map color frame to depth space
    HRESULT hr = coordinateMapper->MapColorFrameToDepthSpace(
        depthSize,
        depthData.data(),
        colorToDepthMap.size(),
        colorToDepthMap.data());

    if (FAILED(hr)) {
        spdlog::error("Failed to map color frame to depth space");
        return false;
    }

    // Process each detected person
    detectedPeople.clear();

    if (!openposeData.contains("people") || openposeData["people"].empty()) {
        return false;
    }

    for (const auto& person : openposeData["people"]) {
        Person3D person3D;

        // Get 2D keypoints from OpenPose
        if (!person.contains("pose_keypoints_2d") || person["pose_keypoints_2d"].empty()) {
            continue;
        }

        const auto& keypoints2d = person["pose_keypoints_2d"];

        // Prepare for depth filtering
        std::vector<float> validDepths;

        // Process each keypoint
        for (size_t i = 0; i < keypoints2d.size() / 3; i++) {
            float x = keypoints2d[i*3];
            float y = keypoints2d[i*3 + 1];
            float confidence = keypoints2d[i*3 + 2];

            // Skip low confidence points
            if (confidence < keypointConfidenceThreshold / 100.0f) {
                Keypoint3D kp = {0, 0, 0, 0};
                person3D.keypoints.push_back(kp);
                continue;
            }

            // Initialize keypoint with original 2D coordinates
            Keypoint3D kp = {x, y, 0, confidence};

            // Check if point is within color image bounds
            if (x >= 0 && x < colorImage.cols && y >= 0 && y < colorImage.rows) {
                // Get corresponding depth point from pre-calculated mapping
                int colorIndex = static_cast<int>(y) * colorImage.cols + static_cast<int>(x);

                if (colorIndex >= 0 && colorIndex < colorToDepthMap.size()) {
                    DepthSpacePoint depthPoint = colorToDepthMap[colorIndex];

                    // Check if mapping is valid
                    if (depthPoint.X >= 0 && depthPoint.Y >= 0) {
                        int depthX = static_cast<int>(depthPoint.X);
                        int depthY = static_cast<int>(depthPoint.Y);

                        // Sample a small region around the point for more robust depth
                        const int windowSize = 3;
                        std::vector<UINT16> depthSamples;

                        for (int dy = -windowSize; dy <= windowSize; dy++) {
                            for (int dx = -windowSize; dx <= windowSize; dx++) {
                                int sx = depthX + dx;
                                int sy = depthY + dy;

                                // Check if within depth image bounds
                                if (sx >= 0 && sx < depthMat16U.cols && sy >= 0 && sy < depthMat16U.rows) {
                                    // Get depth value at this point
                                    UINT16 depth = depthMat16U.at<UINT16>(sy, sx);

                                    // Only consider valid depth values
                                    if (depth > 0) {
                                        depthSamples.push_back(depth);
                                    }
                                }
                            }
                        }

                        // If we have depth samples, use the median for stability
                        if (!depthSamples.empty()) {
                            // Sort and take median
                            std::sort(depthSamples.begin(), depthSamples.end());
                            UINT16 medianDepth = depthSamples[depthSamples.size() / 2];

                            // Convert to camera space
                            CameraSpacePoint cameraPoint;
                            hr = coordinateMapper->MapDepthPointToCameraSpace(
                                depthPoint, medianDepth, &cameraPoint);

                            if (SUCCEEDED(hr)) {
                                // Apply stabilization - only accept reasonable depth values
                                if (cameraPoint.Z > 0.1f && cameraPoint.Z < 5.0f) {
                                    // Update with 3D coordinates
                                    kp.x = cameraPoint.X;
                                    kp.y = cameraPoint.Y;
                                    kp.z = cameraPoint.Z;

                                    // Store valid depths for outlier detection
                                    validDepths.push_back(cameraPoint.Z);
                                }
                            }
                        }
                    }
                }
            }

            // Add keypoint
            person3D.keypoints.push_back(kp);
        }

        // Apply post-processing to smooth and filter keypoints
        if (!person3D.keypoints.empty() && !validDepths.empty()) {
            // Calculate median depth for outlier detection
            std::sort(validDepths.begin(), validDepths.end());
            float medianDepth = validDepths[validDepths.size() / 2];

            // Apply temporal smoothing and outlier rejection
            for (auto& kp : person3D.keypoints) {
                // Skip keypoints with no depth or confidence
                if (kp.confidence <= 0.0f || kp.z <= 0.0f) continue;

                // Reject outliers (more than 50% from median)
                if (std::abs(kp.z - medianDepth) > medianDepth * 0.5f) {
                    // Replace with median depth but keep confidence
                    kp.z = medianDepth;
                }

                // Apply depth-based confidence adjustment
                // Points further away are less reliable
                if (kp.z > 2.0f) {
                    kp.confidence *= (2.0f / kp.z);
                }
            }
        }

        // Add person if they have keypoints
        if (!person3D.keypoints.empty()) {
            detectedPeople.push_back(person3D);
        }
    }

    return !detectedPeople.empty();
}
bool OpenPoseCapture::processFrame(const cv::Mat& colorImage, const cv::Mat& depthImage,
                                 ICoordinateMapper* coordinateMapper,
                                 std::vector<Person3D>& detectedPeople) const {
    if (colorImage.empty() || depthImage.empty() || !coordinateMapper) {
        spdlog::error("Invalid inputs to processFrame");
        return false;
    }

    // Create a unique timestamp for this frame
    auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    std::string imagePath = tempImageDir + "/frame_" + std::to_string(timestamp) + ".png";

    // Ensure temp directory exists
    if (!fs::exists(tempImageDir)) {
        fs::create_directories(tempImageDir);
    }

    // Save the color image to temp directory
    cv::imwrite(imagePath, colorImage);

    // Run OpenPose on the image
    if (!runOpenPoseOnImage(tempImageDir, outputJsonDir)) {
        fs::remove(imagePath);
        return false;
    }

    // Get the latest JSON file generated by OpenPose
    std::string jsonFile = getLastJsonFile(outputJsonDir);
    if (jsonFile.empty()) {
        spdlog::error("No JSON output found from OpenPose");
        fs::remove(imagePath);
        return false;
    }

    // Read and parse the JSON
    json openposeData = readOpenPoseJson(jsonFile);

    // Clean up temp files
    fs::remove(imagePath);
    fs::remove(jsonFile);

    if (openposeData.empty() || !openposeData.contains("people")) {
        spdlog::warn("No people detected in the frame");
        return false;
    }

    // Pre-map the entire color frame to depth space for efficiency
    const int depthWidth = depthImage.cols;
    const int depthHeight = depthImage.rows;
    const int depthSize = depthWidth * depthHeight;

    // Create depth to camera mapping table
    std::vector<DepthSpacePoint> colorToDepthMap(colorImage.cols * colorImage.rows);

    // Get depth frame data
    std::vector<UINT16> depthData(depthSize);
    for (int y = 0; y < depthHeight; y++) {
        for (int x = 0; x < depthWidth; x++) {
            depthData[y * depthWidth + x] = depthImage.at<UINT16>(y, x);
        }
    }

    // Map color frame to depth space
    HRESULT hr = coordinateMapper->MapColorFrameToDepthSpace(
        depthSize,
        depthData.data(),
        colorToDepthMap.size(),
        colorToDepthMap.data());

    if (FAILED(hr)) {
        spdlog::error("Failed to map color frame to depth space");
        return false;
    }

    // Process each detected person
    detectedPeople.clear();
    for (const auto& person : openposeData["people"]) {
        Person3D person3D;

        // Get 2D keypoints from OpenPose (format is [x1,y1,c1,x2,y2,c2,...])
        if (!person.contains("pose_keypoints_2d") || person["pose_keypoints_2d"].empty()) {
            continue;
        }

        const auto& keypoints2d = person["pose_keypoints_2d"];

        // Process each keypoint
        for (size_t i = 0; i < keypoints2d.size() / 3; i++) {
            float x = keypoints2d[i*3];
            float y = keypoints2d[i*3 + 1];
            float confidence = keypoints2d[i*3 + 2];

            // Skip low confidence points
            if (confidence < keypointConfidenceThreshold / 100.0f) {
                Keypoint3D kp = {0, 0, 0, 0};
                person3D.keypoints.push_back(kp);
                continue;
            }

            // Initialize keypoint with original 2D coordinates
            Keypoint3D kp = {x, y, 0, confidence};

            // Check if point is within color image bounds
            if (x >= 0 && x < colorImage.cols && y >= 0 && y < colorImage.rows) {
                // Get corresponding depth point from pre-calculated mapping
                int colorIndex = static_cast<int>(y) * colorImage.cols + static_cast<int>(x);

                if (colorIndex >= 0 && colorIndex < colorToDepthMap.size()) {
                    DepthSpacePoint depthPoint = colorToDepthMap[colorIndex];

                    // Check if mapping is valid (invalid points have negative coordinates)
                    if (depthPoint.X >= 0 && depthPoint.Y >= 0) {
                        int depthX = static_cast<int>(depthPoint.X);
                        int depthY = static_cast<int>(depthPoint.Y);

                        // Check if within depth image bounds
                        if (depthX >= 0 && depthX < depthImage.cols && depthY >= 0 && depthY < depthImage.rows) {
                            // Get depth value at this point
                            UINT16 depth = depthImage.at<UINT16>(depthY, depthX);

                            // Only process valid depth values
                            if (depth > 0) {
                                // Convert to camera (3D) space
                                CameraSpacePoint cameraPoint;
                                hr = coordinateMapper->MapDepthPointToCameraSpace(
                                    depthPoint, depth, &cameraPoint);

                                if (SUCCEEDED(hr)) {
                                    // Update with 3D coordinates
                                    kp.x = cameraPoint.X;
                                    kp.y = cameraPoint.Y;
                                    kp.z = cameraPoint.Z;
                                }
                            }
                        }
                    }
                }
            }

            // Add keypoint (either with 3D coordinates or original 2D if mapping failed)
            person3D.keypoints.push_back(kp);
        }

        // Add person if they have keypoints
        if (!person3D.keypoints.empty()) {
            detectedPeople.push_back(person3D);
        }
    }

    return !detectedPeople.empty();
}

bool OpenPoseCapture::processBatch(const std::vector<std::string>& colorImagePaths,
                                 const std::vector<std::string>& depthRawPaths,
                                 ICoordinateMapper* coordinateMapper,
                                 const std::string& outputDir) const {
    if (colorImagePaths.empty() || colorImagePaths.size() != depthRawPaths.size()) {
        spdlog::error("Invalid batch input");
        return false;
    }

    // Create output directory
    fs::create_directories(outputDir);
    fs::create_directories(outputDir + "/json");
    fs::create_directories(outputDir + "/viz");

    // Create a unique batch directory
    auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    std::string batchTempDir = tempImageDir + "/batch_" + std::to_string(timestamp);
    fs::create_directories(batchTempDir);

    // Copy all images to temp dir for batch processing
    for (size_t i = 0; i < colorImagePaths.size(); i++) {
        // Get the frame index from the filename
        std::string filename = fs::path(colorImagePaths[i]).filename().string();
        fs::copy_file(colorImagePaths[i],
                     batchTempDir + "/" + filename,
                     fs::copy_options::overwrite_existing);
    }

    // Run OpenPose on the batch
    std::string batchJsonDir = outputJsonDir + "/batch_" + std::to_string(timestamp);
    fs::create_directories(batchJsonDir);
    if (!runOpenPoseOnImage(batchTempDir, batchJsonDir)) {
        fs::remove_all(batchTempDir);
        return false;
    }

    // Process each result
    for (size_t i = 0; i < colorImagePaths.size(); i++) {
        // Extract frame index from filename
        std::string filename = fs::path(colorImagePaths[i]).filename().string();
        std::string baseFilename = fs::path(filename).stem().string();

        // Find corresponding JSON file - OpenPose adds '_keypoints' to the filename
        std::string jsonFile = batchJsonDir + "/" + baseFilename + "_keypoints.json";
        if (!fs::exists(jsonFile)) {
            spdlog::warn("No JSON result for frame {}", baseFilename);
            continue;
        }

        // Load depth data
        cv::Mat depthImg = loadRawDepthData(depthRawPaths[i]);
        if (depthImg.empty()) {
            continue;
        }

        // Load color image
        cv::Mat colorImg = cv::imread(colorImagePaths[i]);
        if (colorImg.empty()) {
            continue;
        }

        // Process the frame with 3D lifting
        std::vector<Person3D> people;
        json openposeData = readOpenPoseJson(jsonFile);

        if (process3DLifting(colorImg, depthImg, coordinateMapper, openposeData, people)) {
            // Extract frame index from filename (assuming format frame_X.png)
            size_t underscore = baseFilename.find('_');
            std::string frameIndexStr = (underscore != std::string::npos) ?
                                       baseFilename.substr(underscore + 1) : baseFilename;

            // Save the 3D result
            std::string outJsonPath = outputDir + "/json/frame_" + frameIndexStr + ".json";
            save3DSkeletonToJson(people, outJsonPath);

            // Create visualization
            cv::Mat visualImg = colorImg.clone();
            visualize3DSkeleton(visualImg, people);

            std::string visualPath = outputDir + "/viz/frame_" + frameIndexStr + ".png";
            cv::imwrite(visualPath, visualImg);
        }
    }

    // Clean up temp directories
    fs::remove_all(batchTempDir);
    fs::remove_all(batchJsonDir);

    return true;
}

void OpenPoseCapture::processingWorker(const std::string& outputDir) {
    while (true) {
        ProcessingJob job;
        bool hasJob = false;

        // Get a job from the queue
        {
            std::unique_lock<std::mutex> lock(jobMutex);
            if (!jobQueue.empty()) {
                job = jobQueue.front();
                jobQueue.pop();
                hasJob = true;
            } else if (!isProcessing) {
                // No more jobs and processing has stopped
                break;
            } else {
                // Wait for more jobs
                jobCV.wait(lock);
                continue;
            }
        }

        if (hasJob) {
            try {
                // Load depth data
                cv::Mat depthImg = loadRawDepthData(job.depthPath);
                if (depthImg.empty()) {
                    spdlog::error("Failed to load depth data for frame {}", job.frameIndex);
                    ++jobsCompleted;
                    continue;
                }

                // Load color image
                cv::Mat colorImg = cv::imread(job.colorPath);
                if (colorImg.empty()) {
                    spdlog::error("Failed to load color image for frame {}", job.frameIndex);
                    ++jobsCompleted;
                    continue;
                }

                // Process the frame
                std::vector<Person3D> people;
                {
                    std::lock_guard<std::mutex> lock(coordinateMapperMutex);
                    if (!processFrame(colorImg, depthImg, m_pCoordinateMapper, people)) {
                        spdlog::warn("No people detected in frame {}", job.frameIndex);
                        ++jobsCompleted;
                        continue;
                    }
                }

                // Save the 3D result
                std::string outJsonPath = outputDir + "/json/frame_" + std::to_string(job.frameIndex) + ".json";
                save3DSkeletonToJson(people, outJsonPath);

                // Create visualization
                cv::Mat visualImg = colorImg.clone();
                visualize3DSkeleton(visualImg, people);

                std::string visualPath = outputDir + "/viz/frame_" + std::to_string(job.frameIndex) + ".png";
                cv::imwrite(visualPath, visualImg);

                ++jobsCompleted;

                // Log progress
                if (jobsCompleted % 10 == 0 || jobsCompleted == totalJobs) {
                    spdlog::info("Processed {}/{} frames", jobsCompleted.load(), totalJobs.load());
                }
            } catch (const std::exception& e) {
                spdlog::error("Error processing frame {}: {}", job.frameIndex, e.what());
                ++jobsCompleted;
            }
        }
    }
}

bool OpenPoseCapture::processRecordingDirectory(const std::string& recordingDir,
                                             ICoordinateMapper* coordinateMapper,
                                             const std::string& outputDir,
                                             int numThreads) {
    if (!fs::exists(recordingDir)) {
        spdlog::error("Recording directory does not exist: {}", recordingDir);
        return false;
    }

    if (!coordinateMapper) {
        spdlog::error("Coordinate mapper is null");
        return false;
    }

    // Log the full paths for debugging
    fs::path absRecordingPath = fs::absolute(recordingDir);
    fs::path absOutputPath = fs::absolute(outputDir);

    spdlog::info("Using recording path: {}", absRecordingPath.string());
    spdlog::info("Using output path: {}", absOutputPath.string());

    // Store coordinate mapper for threads to use
    m_pCoordinateMapper = coordinateMapper;

    // Create output directories
    fs::create_directories(outputDir);
    fs::create_directories(outputDir + "/json");
    fs::create_directories(outputDir + "/viz");

    // Check if we have a video file or a directory of images
    bool hasVideoFile = fs::exists(recordingDir + "/color.mp4");
    bool hasColorDir = fs::exists(recordingDir + "/color");
    bool hasDepthDir = fs::exists(recordingDir + "/depth_raw");
    bool hasProcessList = fs::exists(recordingDir + "/process_frames.txt");

    // Verify we have the required directories
    if (!hasDepthDir) {
        spdlog::error("Missing depth_raw directory in {}", recordingDir);
        return false;
    }

    if (!hasVideoFile && !hasColorDir) {
        spdlog::error("Missing color data (neither color.mp4 nor color/ directory found) in {}", recordingDir);
        return false;
    }

    spdlog::info("Found recording with: {}, {}",
                hasVideoFile ? "color.mp4" : "color/ directory",
                hasProcessList ? "process_frames.txt" : "depth_raw/ directory only");

    // Get list of frames to process
    std::vector<int> framesToProcess;
    if (hasProcessList) {
        // Load from the process_frames.txt file
        std::ifstream indexFile(recordingDir + "/process_frames.txt");
        int frameIdx;

        if (!indexFile.is_open()) {
            spdlog::error("Failed to open process_frames.txt");
        } else {
            while (indexFile >> frameIdx) {
                framesToProcess.push_back(frameIdx);
            }
            spdlog::info("Found {} frames to process from index file", framesToProcess.size());
        }
    }

    // If no frames found from process list, or if process list doesn't exist,
    // scan the depth_raw directory
    if (framesToProcess.empty()) {
        spdlog::info("Scanning depth_raw directory for .bin files...");

        try {
            // Count depth files and extract frame numbers
            for (const auto& entry : fs::directory_iterator(recordingDir + "/depth_raw")) {
                if (entry.path().extension() == ".bin") {
                    std::string filename = entry.path().filename().string();
                    // Extract frame number from format "frame_X.bin"
                    size_t underscore = filename.find('_');
                    size_t dot = filename.find('.');
                    if (underscore != std::string::npos && dot != std::string::npos) {
                        int frameIdx = std::stoi(filename.substr(underscore + 1, dot - underscore - 1));
                        framesToProcess.push_back(frameIdx);
                        // Debug output for the first few frames
                        if (framesToProcess.size() <= 5) {
                            spdlog::debug("Found frame file: {} -> Index {}", filename, frameIdx);
                        }
                    }
                }
            }

            // Sort the frame indices
            std::sort(framesToProcess.begin(), framesToProcess.end());

            spdlog::info("Found {} depth frame files", framesToProcess.size());
        } catch (const std::exception& e) {
            spdlog::error("Error scanning depth directory: {}", e.what());
        }
    }

    if (framesToProcess.empty()) {
        spdlog::error("No frames found to process in {}", recordingDir);
        spdlog::error("Please check that the directory contains depth_raw/*.bin files");
        return false;
    }

    // Set performance mode to speed up processing
    bool oldPerformanceMode = performanceMode;
    performanceMode = true;

    // Open video capture if using video format
    cv::VideoCapture videoCapture;
    if (hasVideoFile) {
        std::string videoPath = recordingDir + "/color.mp4";
        spdlog::info("Opening video file: {}", videoPath);
        videoCapture.open(videoPath);
        if (!videoCapture.isOpened()) {
            spdlog::error("Failed to open color video file");
            performanceMode = oldPerformanceMode;
            return false;
        }

        // Check video properties
        int frameWidth = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_WIDTH));
        int frameHeight = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));
        int totalFrames = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_COUNT));

        spdlog::info("Video properties: {}x{}, {} total frames",
                    frameWidth, frameHeight, totalFrames);
    }

    // Set up multi-threading
    int actualThreads = std::min(numThreads, static_cast<int>(std::thread::hardware_concurrency()));
    if (actualThreads < 1) actualThreads = 1;

    // Create a progress report file
    std::string progressFile = outputDir + "/processing_progress.txt";
    std::ofstream progress(progressFile);
    if (progress.is_open()) {
        progress << "Processing started at: " <<
            std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << std::endl;
        progress << "Total frames to process: " << framesToProcess.size() << std::endl;
        progress << "Using " << actualThreads << " threads" << std::endl;
        progress << "Progress:" << std::endl;
        progress.close();
    }

    spdlog::info("Processing {} frames using {} threads", framesToProcess.size(), actualThreads);

    // Initialize worker threads
    isProcessing = true;
    totalJobs = framesToProcess.size();
    jobsCompleted = 0;

    // Process in smaller batches to better utilize OpenPose
    const int batchSize = std::min(20, static_cast<int>(framesToProcess.size()));
    spdlog::info("Using batch size of {} frames for efficiency", batchSize);

    // Create a temporary directory for the batch
    auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    std::string batchTempDir = tempImageDir + "/batch_" + std::to_string(timestamp);
    fs::create_directories(batchTempDir);

    // Create a batch output directory
    std::string batchOutputDir = outputJsonDir + "/batch_" + std::to_string(timestamp);
    fs::create_directories(batchOutputDir);

    auto startTime = std::chrono::high_resolution_clock::now();
    int totalProcessed = 0;

    // Process frames in batches
    for (size_t i = 0; i < framesToProcess.size(); i += batchSize) {
        auto batchStartTime = std::chrono::high_resolution_clock::now();

        // Clear previous batch files
        for (const auto& entry : fs::directory_iterator(batchTempDir)) {
            fs::remove(entry.path());
        }
        for (const auto& entry : fs::directory_iterator(batchOutputDir)) {
            fs::remove(entry.path());
        }

        // Current batch size (might be smaller for the last batch)
        int currentBatchSize = std::min(batchSize, static_cast<int>(framesToProcess.size() - i));
        std::vector<int> batchFrames;

        spdlog::info("Processing batch {}/{} (frames {}-{})",
                    i/batchSize + 1, (framesToProcess.size() + batchSize - 1)/batchSize,
                    i, i + currentBatchSize - 1);

        // Extract batch frames
        for (int j = 0; j < currentBatchSize; j++) {
            int frameIdx = framesToProcess[i + j];
            batchFrames.push_back(frameIdx);

            std::string colorPath;

            if (hasVideoFile) {
                // Extract frame from video
                videoCapture.set(cv::CAP_PROP_POS_FRAMES, frameIdx);
                cv::Mat frame;
                if (videoCapture.read(frame)) {
                    colorPath = batchTempDir + "/frame_" + std::to_string(frameIdx) + ".png";
                    cv::imwrite(colorPath, frame);
                } else {
                    spdlog::error("Failed to read frame {} from video", frameIdx);
                    continue;
                }
            } else {
                // Copy from color directory
                std::string sourcePath = recordingDir + "/color/frame_" + std::to_string(frameIdx) + ".png";
                colorPath = batchTempDir + "/frame_" + std::to_string(frameIdx) + ".png";

                if (fs::exists(sourcePath)) {
                    fs::copy_file(sourcePath, colorPath, fs::copy_options::overwrite_existing);
                } else {
                    spdlog::error("Color image not found: {}", sourcePath);
                    continue;
                }
            }
        }

        // Run OpenPose on the batch
        if (!runOpenPoseOnImage(batchTempDir, batchOutputDir)) {
            spdlog::error("Failed to process batch with OpenPose");
            continue;
        }

        // Process each result in the batch with ThreadPool
        std::vector<std::thread> batchThreads;
        std::mutex resultsMutex;

        for (int frameIdx : batchFrames) {
            // Find the JSON output
            std::string jsonFilename = "frame_" + std::to_string(frameIdx) + "_keypoints.json";
            std::string jsonPath = batchOutputDir + "/" + jsonFilename;

            if (!fs::exists(jsonPath)) {
                // Try alternative naming (OpenPose sometimes uses different formats)
                for (const auto& entry : fs::directory_iterator(batchOutputDir)) {
                    if (entry.path().string().find(std::to_string(frameIdx)) != std::string::npos) {
                        jsonPath = entry.path().string();
                        break;
                    }
                }
            }

            if (!fs::exists(jsonPath)) {
                spdlog::warn("No OpenPose output found for frame {}", frameIdx);
                continue;
            }

            // Load depth data
            std::string depthPath = recordingDir + "/depth_raw/frame_" + std::to_string(frameIdx) + ".bin";
            if (!fs::exists(depthPath)) {
                spdlog::warn("Depth data not found for frame {}", frameIdx);
                continue;
            }

            // Create a thread to process this frame
            if (batchThreads.size() < actualThreads) {
                batchThreads.emplace_back([this, &resultsMutex, frameIdx, jsonPath, depthPath, &batchTempDir, &recordingDir, &outputDir]() {
                    try {
                        // Load depth image
                        cv::Mat depthImg = loadRawDepthData(depthPath);
                        if (depthImg.empty()) {
                            spdlog::error("Failed to load depth data for frame {}", frameIdx);
                            return;
                        }

                        // Load color image
                        std::string colorPath = batchTempDir + "/frame_" + std::to_string(frameIdx) + ".png";
                        cv::Mat colorImg = cv::imread(colorPath);
                        if (colorImg.empty()) {
                            spdlog::error("Failed to load color image for frame {}", frameIdx);
                            return;
                        }

                        // Parse JSON
                        json openposeData = readOpenPoseJson(jsonPath);
                        if (openposeData.empty() || !openposeData.contains("people")) {
                            spdlog::warn("No people detected in frame {}", frameIdx);
                            return;
                        }

                        // Process 3D lifting
                        std::vector<Person3D> people;
                        {
                            std::lock_guard<std::mutex> lock(coordinateMapperMutex);
                            if (!process3DLifting(colorImg, depthImg, m_pCoordinateMapper, openposeData, people)) {
                                spdlog::warn("3D lifting failed for frame {}", frameIdx);
                                return;
                            }
                        }

                        // Save results
                        {
                            std::lock_guard<std::mutex> lock(resultsMutex);

                            // Save JSON
                            std::string outJsonPath = outputDir + "/json/frame_" + std::to_string(frameIdx) + ".json";
                            if (!save3DSkeletonToJson(people, outJsonPath)) {
                                spdlog::error("Failed to save JSON for frame {}", frameIdx);
                            }

                            // Create visualization
                            cv::Mat visualImg = colorImg.clone();
                            visualize3DSkeleton(visualImg, people);

                            std::string visualPath = outputDir + "/viz/frame_" + std::to_string(frameIdx) + ".png";
                            cv::imwrite(visualPath, visualImg);

                            // Update progress file
                            std::ofstream progress(outputDir + "/processing_progress.txt", std::ios::app);
                            if (progress.is_open()) {
                                progress << "Processed frame " << frameIdx << std::endl;
                                progress.close();
                            }
                        }
                    } catch (const std::exception& e) {
                        spdlog::error("Error processing frame {}: {}", frameIdx, e.what());
                    }
                });
            } else {
                // Wait for a thread to complete if we've reached the thread limit
                if (!batchThreads.empty()) {
                    batchThreads[0].join();
                    batchThreads.erase(batchThreads.begin());
                }

                // Now create a new thread
                batchThreads.emplace_back([this, &resultsMutex, frameIdx, jsonPath, depthPath, &batchTempDir, &recordingDir, &outputDir]() {
                    // Same processing code as above
                    try {
                        // Load depth image
                        cv::Mat depthImg = loadRawDepthData(depthPath);
                        if (depthImg.empty()) {
                            spdlog::error("Failed to load depth data for frame {}", frameIdx);
                            return;
                        }

                        // Load color image
                        std::string colorPath = batchTempDir + "/frame_" + std::to_string(frameIdx) + ".png";
                        cv::Mat colorImg = cv::imread(colorPath);
                        if (colorImg.empty()) {
                            spdlog::error("Failed to load color image for frame {}", frameIdx);
                            return;
                        }

                        // Parse JSON
                        json openposeData = readOpenPoseJson(jsonPath);
                        if (openposeData.empty() || !openposeData.contains("people")) {
                            spdlog::warn("No people detected in frame {}", frameIdx);
                            return;
                        }

                        // Process 3D lifting
                        std::vector<Person3D> people;
                        {
                            std::lock_guard<std::mutex> lock(coordinateMapperMutex);
                            if (!process3DLifting(colorImg, depthImg, m_pCoordinateMapper, openposeData, people)) {
                                spdlog::warn("3D lifting failed for frame {}", frameIdx);
                                return;
                            }
                        }

                        // Save results
                        {
                            std::lock_guard<std::mutex> lock(resultsMutex);

                            // Save JSON
                            std::string outJsonPath = outputDir + "/json/frame_" + std::to_string(frameIdx) + ".json";
                            if (!save3DSkeletonToJson(people, outJsonPath)) {
                                spdlog::error("Failed to save JSON for frame {}", frameIdx);
                            }

                            // Create visualization
                            cv::Mat visualImg = colorImg.clone();
                            visualize3DSkeleton(visualImg, people);

                            std::string visualPath = outputDir + "/viz/frame_" + std::to_string(frameIdx) + ".png";
                            cv::imwrite(visualPath, visualImg);

                            // Update progress file
                            std::ofstream progress(outputDir + "/processing_progress.txt", std::ios::app);
                            if (progress.is_open()) {
                                progress << "Processed frame " << frameIdx << std::endl;
                                progress.close();
                            }
                        }
                    } catch (const std::exception& e) {
                        spdlog::error("Error processing frame {}: {}", frameIdx, e.what());
                    }
                });
            }

            totalProcessed++;
        }

        // Wait for all threads in this batch to complete
        for (auto& thread : batchThreads) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        // Report batch timing
        auto batchEndTime = std::chrono::high_resolution_clock::now();
        auto batchDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
            batchEndTime - batchStartTime).count();

        spdlog::info("Batch {}/{} completed in {:.2f} seconds ({:.2f} seconds per frame)",
                    i/batchSize + 1, (framesToProcess.size() + batchSize - 1)/batchSize,
                    batchDuration / 1000.0,
                    batchDuration / 1000.0 / currentBatchSize);

        // Update overall progress
        double progress_percentage = 100.0 * totalProcessed / framesToProcess.size();

        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsedMillis = std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - startTime).count();

        double framesPerSecond = static_cast<double>(totalProcessed) / (elapsedMillis / 1000.0);
        double estimatedTotalSeconds = framesToProcess.size() / framesPerSecond;
        double remainingSeconds = estimatedTotalSeconds - (elapsedMillis / 1000.0);

        spdlog::info("Progress: {:.1f}% ({}/{} frames) - {:.2f} frames/sec - Est. {:.0f}s remaining",
                    progress_percentage, totalProcessed, framesToProcess.size(),
                    framesPerSecond, remainingSeconds);
    }

    // Clean up temporary directories
    try {
        if (fs::exists(batchTempDir)) {
            fs::remove_all(batchTempDir);
        }
        if (fs::exists(batchOutputDir)) {
            fs::remove_all(batchOutputDir);
        }
    } catch (const std::exception& e) {
        spdlog::warn("Error cleaning up temporary directories: {}", e.what());
    }

    // Generate a summary video from visualizations
    try {
        std::vector<std::string> vizImages;
        for (const auto& entry : fs::directory_iterator(outputDir + "/viz")) {
            if (entry.path().extension() == ".png") {
            vizImages.push_back(entry.path().string());
            }
        }

        if (!vizImages.empty()) {
            // Sort by frame number
            std::sort(vizImages.begin(), vizImages.end());

        // Create video writer
        cv::VideoWriter vizWriter;
        std::string vizVideoPath = outputDir + "/visualization.mp4";

        // Read first image to get dimensions
        cv::Mat firstImage = cv::imread(vizImages[0]);
        if (!firstImage.empty()) {
            int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

            // Try to read the actual FPS from metadata file
            double calculatedFps = 5.0; // Default fallback FPS
            std::string metadataPath = recordingDir + "/metadata.txt";

            if (fs::exists(metadataPath)) {
                try {
                    std::ifstream metaFile(metadataPath);
                    std::string line;

                    // Search for "Actual FPS:" in metadata
                    while (std::getline(metaFile, line)) {
                        if (line.find("Actual FPS:") != std::string::npos) {
                            // Extract FPS value
                            size_t pos = line.find(":");
                            if (pos != std::string::npos) {
                                std::string fpsStr = line.substr(pos + 1);
                                try {
                                    calculatedFps = std::stod(fpsStr);
                                    spdlog::info("Found FPS in metadata: {:.2f}", calculatedFps);
                                    break;
                                } catch (...) {
                                    // If parsing fails, use default
                                }
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    spdlog::warn("Error reading FPS from metadata: {}", e.what());
                }
            }

            // If we couldn't get FPS from metadata, try to calculate from frame timestamps
            if (calculatedFps <= 0.1) {
                try {
                    // Option 1: Try to calculate from process_frames.txt timestamps if available
                    std::string timeStampFile = recordingDir + "/frame_timestamps.csv";
                    if (fs::exists(timeStampFile)) {
                        std::ifstream timestampFile(timeStampFile);
                        std::string line;
                        std::getline(timestampFile, line); // Skip header

                        uint64_t firstTs = 0, lastTs = 0;
                        int frameCount = 0;

                        while (std::getline(timestampFile, line)) {
                            std::istringstream iss(line);
                            std::string token;

                            std::getline(iss, token, ','); // frame index
                            std::getline(iss, token, ','); // timestamp

                            uint64_t ts = std::stoull(token);

                            if (frameCount == 0) {
                                firstTs = ts;
                            }
                            lastTs = ts;
                            frameCount++;
                        }

                        if (frameCount >= 2) {
                            double durationSec = (lastTs - firstTs) / 1000000000.0; // Convert to seconds
                            if (durationSec > 0) {
                                calculatedFps = frameCount / durationSec;
                                spdlog::info("Calculated FPS from timestamps: {:.2f}", calculatedFps);
                            }
                        }
                    }

                    // Option 2: If that fails, try to use file timestamps from the PNG frames
                    if (calculatedFps <= 0.1 && !vizImages.empty() && vizImages.size() >= 2) {
                        auto firstTime = fs::last_write_time(vizImages.front());
                        auto lastTime = fs::last_write_time(vizImages.back());

                        // Convert to duration
                        auto duration = lastTime - firstTime;
                        auto durationSec = std::chrono::duration_cast<std::chrono::seconds>(duration).count();

                        if (durationSec > 0) {
                            calculatedFps = vizImages.size() / static_cast<double>(durationSec);
                            spdlog::info("Calculated FPS from file timestamps: {:.2f}", calculatedFps);
                        }
                    }
                } catch (const std::exception& e) {
                    spdlog::warn("Error calculating FPS: {}", e.what());
                }
            }

            // Option 3: Check the process_frames.txt to determine time differences between frames
            if (calculatedFps <= 0.1) {
                try {
                    std::string processFramesPath = recordingDir + "/process_frames.txt";
                    if (fs::exists(processFramesPath) && fs::file_size(processFramesPath) > 0) {
                        // Read the frame indices to process
                        std::vector<int> frameIndices;
                        std::ifstream processFramesFile(processFramesPath);
                        int frameIdx;
                        while (processFramesFile >> frameIdx) {
                            frameIndices.push_back(frameIdx);
                        }

                        if (frameIndices.size() >= 2) {
                            // Calculate frame rate from number of frames and metadata duration
                            std::string metadataPath = recordingDir + "/recording_summary.txt";
                            if (fs::exists(metadataPath)) {
                                std::ifstream metaFile(metadataPath);
                                std::string line;

                                double durationSec = 0.0;
                                while (std::getline(metaFile, line)) {
                                    if (line.find("Recording duration (sec):") != std::string::npos) {
                                        size_t pos = line.find(":");
                                        if (pos != std::string::npos) {
                                            std::string durStr = line.substr(pos + 1);
                                            try {
                                                durationSec = std::stod(durStr);
                                                break;
                                            } catch (...) {
                                                // If parsing fails, continue
                                            }
                                        }
                                    }
                                }

                                if (durationSec > 0) {
                                    calculatedFps = frameIndices.size() / durationSec;
                                    spdlog::info("Calculated FPS from recording duration: {:.2f}", calculatedFps);
                                }
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    spdlog::warn("Error calculating FPS from process_frames.txt: {}", e.what());
                }
            }

            // Ensure a reasonable FPS range (not too low or too high)
            calculatedFps = std::max(1.0, std::min(calculatedFps, 30.0));

            spdlog::info("Creating visualization video from {} frames at {:.2f} FPS...",
                        vizImages.size(), calculatedFps);

            vizWriter.open(vizVideoPath, fourcc, calculatedFps, firstImage.size());

            if (vizWriter.isOpened()) {
                for (const auto& imgPath : vizImages) {
                    cv::Mat image = cv::imread(imgPath);
                    if (!image.empty()) {
                        vizWriter.write(image);
                    }
                }

                vizWriter.release();
                spdlog::info("Created visualization video at {}", vizVideoPath);
            }
        }
    }
} catch (const std::exception& e) {
    spdlog::error("Error creating visualization video: {}", e.what());
}

    // Report final timing
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::seconds>(
        endTime - startTime).count();

    std::ofstream finalReport(outputDir + "/processing_summary.txt");
    if (finalReport.is_open()) {
        finalReport << "Processing completed at: " <<
            std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << std::endl;
        finalReport << "Total frames processed: " << totalProcessed << std::endl;
        finalReport << "Total processing time: " << totalDuration << " seconds" << std::endl;
        finalReport << "Average time per frame: " <<
            static_cast<double>(totalDuration) / totalProcessed << " seconds" << std::endl;
        finalReport << "Frames per second: " <<
            static_cast<double>(totalProcessed) / totalDuration << std::endl;
        finalReport.close();
    }

    spdlog::info("Completed processing all frames");
    spdlog::info("Total time: {} seconds (avg {:.2f} seconds/frame)",
                totalDuration, static_cast<double>(totalDuration) / totalProcessed);

    // Restore previous performance mode setting
    performanceMode = oldPerformanceMode;

    if (hasVideoFile) {
        videoCapture.release();
    }

    return true;
}

bool OpenPoseCapture::save3DSkeletonToJson(const std::vector<Person3D>& people, const std::string& outputPath) {
    json output;
    output["people"] = json::array();

    for (const auto& person : people) {
        json personJson;
        json keypointsJson = json::array();

        for (const auto& kp : person.keypoints) {
            keypointsJson.push_back(kp.x);
            keypointsJson.push_back(kp.y);
            keypointsJson.push_back(kp.z);
            keypointsJson.push_back(kp.confidence);
        }

        personJson["pose_keypoints_3d"] = keypointsJson;
        output["people"].push_back(personJson);
    }

    try {
        std::ofstream file(outputPath);
        file << output.dump(4);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to save 3D skeleton: {}", e.what());
        return false;
    }
}

void OpenPoseCapture::visualize3DSkeleton(cv::Mat& image, const std::vector<Person3D>& people) {
    // Define the skeleton connections for visualization
    const std::vector<std::pair<int, int>> connections = {
        {0, 1}, {1, 2}, {1, 5}, {1, 8}, {2, 3}, {3, 4}, {5, 6},
        {6, 7}, {8, 9}, {8, 12}, {9, 10}, {10, 11}, {12, 13}, {13, 14}
    };

    // Add a title with frame information
    cv::putText(image, "3D Skeleton Reconstruction",
               cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
               cv::Scalar(0, 0, 255), 2);

    for (const auto& person : people) {
        const auto& keypoints = person.keypoints;

        // Draw keypoints
        for (size_t i = 0; i < keypoints.size(); i++) {
            const auto& kp = keypoints[i];

            // Skip invalid points
            if (kp.confidence < 0.1f) continue;

            // Draw the keypoint - the z value affects the color (red to blue)
            // Normalize z between 0 and 1 for visualization
            float normalizedZ = kp.z;
            if (normalizedZ > 5.0f) normalizedZ = 5.0f;
            normalizedZ = normalizedZ / 5.0f;

            cv::Scalar color(255 * normalizedZ, 0, 255 * (1.0f - normalizedZ));
            cv::circle(image, cv::Point(kp.x, kp.y), 5, color, -1);

            // Draw keypoint ID and depth value
            std::stringstream ss;
            ss << i << ":" << std::fixed << std::setprecision(2) << kp.z << "m";
            cv::putText(image, ss.str(), cv::Point(kp.x + 10, kp.y),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
        }

        // Draw connections
        for (const auto& connection : connections) {
            int start = connection.first;
            int end = connection.second;

            if (start >= keypoints.size() || end >= keypoints.size()) {
                continue;
            }

            const auto& kp1 = keypoints[start];
            const auto& kp2 = keypoints[end];

            if (kp1.confidence < 0.1f || kp2.confidence < 0.1f) {
                continue;
            }

            // Average z value for the line color
            float avgZ = (kp1.z + kp2.z) / 2.0f;
            if (avgZ > 5.0f) avgZ = 5.0f;
            avgZ = avgZ / 5.0f;

            cv::Scalar color(255 * avgZ, 0, 255 * (1.0f - avgZ));
            cv::line(image, cv::Point(kp1.x, kp1.y), cv::Point(kp2.x, kp2.y), color, 2);
        }
    }

    // Add a depth scale bar
    int scaleWidth = 30;
    int scaleHeight = 200;
    int scaleX = image.cols - scaleWidth - 20;
    int scaleY = 50;

    // Draw depth scale gradient
    for (int y = 0; y < scaleHeight; y++) {
        float normalizedZ = 1.0f - (float)y / scaleHeight;
        cv::Scalar color(255 * normalizedZ, 0, 255 * (1.0f - normalizedZ));
        cv::rectangle(image,
                     cv::Point(scaleX, scaleY + y),
                     cv::Point(scaleX + scaleWidth, scaleY + y + 1),
                     color, -1);
    }

    // Add labels to scale
    cv::putText(image, "0m", cv::Point(scaleX + scaleWidth + 5, scaleY + scaleHeight),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    cv::putText(image, "5m", cv::Point(scaleX + scaleWidth + 5, scaleY),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    cv::putText(image, "Depth", cv::Point(scaleX, scaleY - 20),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}