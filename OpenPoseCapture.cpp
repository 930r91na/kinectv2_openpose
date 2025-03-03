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
        << " --render_pose 0";

    // Add accuracy settings
    if (useMaximumAccuracy) {
        cmd << " --net_resolution 1312x736"
            << " --scale_number 4"
            << " --scale_gap 0.25";
    } else {
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

                        // Check if within depth image bounds
                        if (depthX >= 0 && depthX < depthMat16U.cols &&
                            depthY >= 0 && depthY < depthMat16U.rows) {

                            // Get depth value at this point
                            UINT16 depth = depthMat16U.at<UINT16>(depthY, depthX);

                            // Only process valid depth values
                            if (depth > 0) {
                                // Convert to camera space
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

            // Add keypoint
            person3D.keypoints.push_back(kp);
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

    // Store coordinate mapper for threads to use
    m_pCoordinateMapper = coordinateMapper;

    // Create output directories
    fs::create_directories(outputDir);
    fs::create_directories(outputDir + "/json");
    fs::create_directories(outputDir + "/viz");

    // Check if we have a video file or a directory of images
    bool hasVideoFile = fs::exists(recordingDir + "/color.mp4");
    bool hasColorDir = fs::exists(recordingDir + "/color");
    bool hasProcessList = fs::exists(recordingDir + "/process_frames.txt");

    // Get list of frames to process
    std::vector<int> framesToProcess;
    if (hasProcessList) {
        // Load from the process_frames.txt file
        std::ifstream indexFile(recordingDir + "/process_frames.txt");
        int frameIdx;
        while (indexFile >> frameIdx) {
            framesToProcess.push_back(frameIdx);
        }
        spdlog::info("Found {} frames to process from index file", framesToProcess.size());
    } else {
        // Just count depth files
        for (const auto& entry : fs::directory_iterator(recordingDir + "/depth_raw")) {
            if (entry.path().extension() == ".bin") {
                std::string filename = entry.path().filename().string();
                // Extract frame number from format "frame_X.bin"
                size_t underscore = filename.find('_');
                size_t dot = filename.find('.');
                if (underscore != std::string::npos && dot != std::string::npos) {
                    int frameIdx = std::stoi(filename.substr(underscore + 1, dot - underscore - 1));
                    framesToProcess.push_back(frameIdx);
                }
            }
        }
        std::ranges::sort(framesToProcess);
        spdlog::info("Found {} depth frame files", framesToProcess.size());
    }

    if (framesToProcess.empty()) {
        spdlog::error("No frames found to process");
        return false;
    }

    // Open video capture if using video format
    cv::VideoCapture videoCapture;
    if (hasVideoFile) {
        videoCapture.open(recordingDir + "/color.mp4");
        if (!videoCapture.isOpened()) {
            spdlog::error("Failed to open color video file");
            return false;
        }
    }

    // Set up multi-threading
    int actualThreads = std::min(numThreads, static_cast<int>(std::thread::hardware_concurrency()));
    if (actualThreads < 1) actualThreads = 1;

    spdlog::info("Processing {} frames using {} threads", framesToProcess.size(), actualThreads);

    // Initialize worker threads
    isProcessing = true;
    totalJobs = framesToProcess.size();
    jobsCompleted = 0;

    for (int i = 0; i < actualThreads; i++) {
        workerThreads.emplace_back(&OpenPoseCapture::processingWorker, this, outputDir);
    }

    // Add jobs to the queue
    for (int frameIdx : framesToProcess) {
        ProcessingJob job;
        job.frameIndex = frameIdx;
        job.depthPath = recordingDir + "/depth_raw/frame_" + std::to_string(frameIdx) + ".bin";

        if (hasVideoFile) {
            // Create a temporary file from the video frame
            std::string tempColorPath = tempImageDir + "/video_frame_" + std::to_string(frameIdx) + ".png";
            videoCapture.set(cv::CAP_PROP_POS_FRAMES, frameIdx);

            cv::Mat frame;
            if (videoCapture.read(frame)) {
                cv::imwrite(tempColorPath, frame);
                job.colorPath = tempColorPath;
            } else {
                spdlog::error("Failed to read frame {} from video", frameIdx);
                continue;
            }
        } else {
            job.colorPath = recordingDir + "/color/frame_" + std::to_string(frameIdx) + ".png";
        }

        // Add to queue
        {
            std::unique_lock<std::mutex> lock(jobMutex);
            jobQueue.push(job);
        }
        jobCV.notify_one();
    }

    // Signal completion
    isProcessing = false;
    jobCV.notify_all();

    // Wait for all threads to finish
    for (auto& thread : workerThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Clean up
    workerThreads.clear();
    if (hasVideoFile) {
        videoCapture.release();
    }

    spdlog::info("Completed processing all frames");
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

            // Draw depth value
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << kp.z << "m";
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
}