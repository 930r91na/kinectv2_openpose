// ReSharper disable CppDFAUnusedValue
#include "KinectDepthChecker.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <chrono>
#include <iostream>
#include <thread>

void KinectDepthChecker::setupLogger() {
    try {
        // Create console sink
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::info);

        // Create file sink
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("kinect_log.txt", true);
        file_sink->set_level(spdlog::level::trace);

        // Create logger with both sinks
        std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};
        logger = std::make_shared<spdlog::logger>("kinect", sinks.begin(), sinks.end());
    	logger->set_level(spdlog::level::debug);

        // Set pattern to include milliseconds
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
        logger->set_level(spdlog::level::trace);

        register_logger(logger);

    	logger->flush_on(spdlog::level::debug);
        logger->info("Logger initialized");
    }
    catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
        throw;
    }
}

cv::Mat KinectDepthChecker::getColorImage() const {
	if (colorImg.empty()) {
		return cv::Mat();
	}

	return colorImg.clone();
}

cv::Mat KinectDepthChecker::getDepthImage() const {
	if (rawDepthImg.empty()) {
		return cv::Mat();
	}
	return rawDepthImg.clone();
}

KinectDepthChecker::KinectDepthChecker() :
    m_pKinectSensor(nullptr),
    m_pDepthFrameReader(nullptr),
    m_pCoordinateMapper(nullptr),
    m_pBodyFrameReader(nullptr),
    m_pBodyIndexFrameReader(nullptr),
	m_pColorFrameReader(nullptr),
	showWindows(false),
    initialized(false) {
    setupLogger();
    logger->info("KinectDepthChecker initialized");
}

KinectDepthChecker::~KinectDepthChecker() {
    SafeRelease(m_pDepthFrameReader);
    SafeRelease(m_pBodyFrameReader);
    SafeRelease(m_pBodyIndexFrameReader);
    SafeRelease(m_pCoordinateMapper);
	SafeRelease(m_pBodyFrameReader);
	SafeRelease(m_pColorFrameReader);

    if (m_pKinectSensor) {
        m_pKinectSensor->Close();
        SafeRelease(m_pKinectSensor);
    }
    logger->info("KinectDepthChecker destroyed");
}

bool KinectDepthChecker::initialize() {
    logger->info("Starting Kinect initialization");

    HRESULT hr = GetDefaultKinectSensor(&m_pKinectSensor);
    if (FAILED(hr) || !m_pKinectSensor) {
        logger->error("Failed to get Kinect sensor");
        return false;
    }

    hr = m_pKinectSensor->Open();
    if (FAILED(hr)) {
        logger->error("Failed to open sensor");
        return false;
    }

    // Get the coordinate mapper
    hr = m_pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
    if (FAILED(hr)) {
        logger->error("Failed to get coordinate mapper");
        return false;
    }

    // Get depth reader
    IDepthFrameSource* depthFrameSource = nullptr;
    hr = m_pKinectSensor->get_DepthFrameSource(&depthFrameSource);
    if (SUCCEEDED(hr)) {
        hr = depthFrameSource->OpenReader(&m_pDepthFrameReader);
        SafeRelease(depthFrameSource);
    }

	// Get color reader
	IColorFrameSource* colorFrameSource = nullptr;
	hr = m_pKinectSensor->get_ColorFrameSource(&colorFrameSource);
	if (SUCCEEDED(hr)) {
		hr = colorFrameSource->OpenReader(&m_pColorFrameReader);
		if (FAILED(hr)) {
			logger->error("Failed to create color frame reader");
		} else {
			logger->info("Color frame reader created successfully");
		}
		SafeRelease(colorFrameSource);
	}

    // Get body reader
    IBodyFrameSource* bodyFrameSource = nullptr;
    hr = m_pKinectSensor->get_BodyFrameSource(&bodyFrameSource);
    if (SUCCEEDED(hr)) {
        hr = bodyFrameSource->OpenReader(&m_pBodyFrameReader);
        SafeRelease(bodyFrameSource);
    }

    // Get body index reader
    IBodyIndexFrameSource* bodyIndexFrameSource = nullptr;
    hr = m_pKinectSensor->get_BodyIndexFrameSource(&bodyIndexFrameSource);
    if (SUCCEEDED(hr)) {
        hr = bodyIndexFrameSource->OpenReader(&m_pBodyIndexFrameReader);
        SafeRelease(bodyIndexFrameSource);

    	if (FAILED(hr)) {
    		logger->error("Failed to create body frame reader");
    		return false;
    	}
    	logger->info("Body frame reader created successfully");
    }

    // Initialize Mat objects
    skeletonImg = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC3);
    depthImg = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC1);
	colorImg = cv::Mat(cColorHeight, cColorWidth, CV_8UC4);

    initialized = true;
    logger->info("Kinect initialization completed successfully");
    return true;
}

void KinectDepthChecker::update(bool visualize) {
    static int frameCount = 0;
    frameCount++;

    // Start with a clean slate
    skeletonImg = cv::Mat::zeros(cDepthHeight, cDepthWidth, CV_8UC3);

    // Process depth frame first
    IDepthFrame* pDepthFrame = nullptr;
    HRESULT hr = m_pDepthFrameReader->AcquireLatestFrame(&pDepthFrame);
    if (SUCCEEDED(hr)) {
        UINT16* depthArray = new UINT16[cDepthHeight * cDepthWidth];
        pDepthFrame->CopyFrameDataToArray(cDepthHeight * cDepthWidth, depthArray);
        rawDepthImg = cv::Mat(cDepthHeight, cDepthWidth, CV_16UC1, depthArray).clone();

        // Normalize depth data for better visualization
        cv::Mat depthMat(cDepthHeight, cDepthWidth, CV_16UC1, depthArray);
        cv::normalize(depthMat, depthImg, 0, 255, cv::NORM_MINMAX, CV_8UC1);

        delete[] depthArray;
        SafeRelease(pDepthFrame);
    }

    // Process color frame
    IColorFrame* pColorFrame = nullptr;
    hr = m_pColorFrameReader->AcquireLatestFrame(&pColorFrame);
    if (SUCCEEDED(hr)) {
        // Get color frame description
        IFrameDescription* colorFrameDescription = nullptr;
        hr = pColorFrame->get_FrameDescription(&colorFrameDescription);

        if (SUCCEEDED(hr)) {
            int width = 0, height = 0;
            colorFrameDescription->get_Width(&width);
            colorFrameDescription->get_Height(&height);

            // Make sure our Mat is the right size
            if (colorImg.cols != width || colorImg.rows != height) {
                colorImg = cv::Mat(height, width, CV_8UC4);
            }

            // Copy the color data to our Mat
            hr = pColorFrame->CopyConvertedFrameDataToArray(
                width * height * 4,  // 4 bytes per pixel (BGRA)
                reinterpret_cast<BYTE*>(colorImg.data),
                ColorImageFormat_Bgra
            );

            if (SUCCEEDED(hr)) {
                logger->debug("Color frame acquired successfully");
            } else {
                logger->error("Failed to copy color frame data");
            }

            SafeRelease(colorFrameDescription);
        }

        SafeRelease(pColorFrame);
    }

    // Process body frame
    IBodyFrame* pBodyFrame = nullptr;
    hr = m_pBodyFrameReader->AcquireLatestFrame(&pBodyFrame);
    if (SUCCEEDED(hr)) {
        IBody* ppBodies[BODY_COUNT] = { nullptr };
        hr = pBodyFrame->GetAndRefreshBodyData(_countof(ppBodies), ppBodies);

        if (SUCCEEDED(hr)) {
            logger->debug("Processing body frame");
            ProcessBody(BODY_COUNT, ppBodies);
        }

        for (int i = 0; i < _countof(ppBodies); ++i) {
            SafeRelease(ppBodies[i]);
        }
        SafeRelease(pBodyFrame);
    }

    // Draw frame info
    cv::putText(skeletonImg, "Frame: " + std::to_string(frameCount),
                cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(255,255,255), 2);

    // Only display images if visualization is enabled
    if (visualize && showWindows) {
        try {
            cv::imshow("Depth Image", depthImg);
            cv::imshow("Skeleton", skeletonImg);
        } catch (const cv::Exception& e) {
            logger->error("OpenCV error in visualization: {}", e.what());
        }
    }
}

void KinectDepthChecker::checkDepthFPS(int durationSeconds) const {
    if (!initialized) {
        logger->error("Device not initialized!");
        return;
    }

    int frameCount = 0;
    int failedFrames = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    logger->info("Starting FPS check for {} seconds", durationSeconds);
    logger->info("Capturing frames...");

    while (true) {
        IDepthFrame* frame = nullptr;
        HRESULT hr = m_pDepthFrameReader->AcquireLatestFrame(&frame);

        if (SUCCEEDED(hr) && frame) {
            frameCount++;
            frame->Release();
            if (frameCount % 30 == 0) { // Log every 30 frames
                logger->debug("Captured {} frames", frameCount);
            }
        } else {
            failedFrames++;
            logger->trace("Frame acquisition failed, HRESULT: {:x}", hr);
        }

        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>
            (currentTime - startTime).count();

        if (elapsedSeconds >= durationSeconds) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    float fps = static_cast<float>(frameCount) / durationSeconds;

    logger->info("\nFPS Check Results:");
    logger->info("------------------");
    logger->info("Frames captured: {}", frameCount);
    logger->info("Failed frame reads: {}", failedFrames);
    logger->info("Average FPS: {:.1f}", fps);

    // Get depth camera resolution
    IDepthFrameSource* depthSource = nullptr;
    if (SUCCEEDED(m_pKinectSensor->get_DepthFrameSource(&depthSource))) {
        IFrameDescription* frameDescription = nullptr;
        if (SUCCEEDED(depthSource->get_FrameDescription(&frameDescription))) {
            int width = 0, height = 0;
            frameDescription->get_Width(&width);
            frameDescription->get_Height(&height);
            logger->info("Depth camera resolution: {}x{}", width, height);
            frameDescription->Release();
        }
        depthSource->Release();
    }

    if (fps < 1.0f) {
        logger->error("No meaningful frames captured - check Kinect connection");
    } else if (fps < 25.0f) {
        logger->warn("Frame rate below optimal (expected 30 FPS)");
    } else {
        logger->info("Frame rate within normal range");
    }
}

void KinectDepthChecker::ProcessBody(int nBodyCount, IBody **ppBodies) {
	int trackedBodies = 0;
	logger->debug("Starting to process {} potential bodies", nBodyCount);

	for (int i = 0; i < nBodyCount; ++i)
	{
		if (IBody* pBody = ppBodies[i])
		{
			BOOLEAN bTracked = false;

			if (HRESULT hr = pBody->get_IsTracked(&bTracked); SUCCEEDED(hr) && bTracked)
			{
				trackedBodies++;
				logger->debug("Body {} is actually tracked", i);
				Joint joints[JointType_Count];
				HandState leftHandState = HandState_Unknown;
				HandState rightHandState = HandState_Unknown;

				pBody->get_HandLeftState(&leftHandState);
				pBody->get_HandRightState(&rightHandState);

				DepthSpacePoint *depthSpacePosition = new DepthSpacePoint[_countof(joints)];

				hr = pBody->GetJoints(_countof(joints), joints);
				if (SUCCEEDED(hr))
				{
					logger->debug("Successfully got joints for body {}", i);
					for (int j = 0; j < _countof(joints); ++j)
					{
						m_pCoordinateMapper->MapCameraPointToDepthSpace(joints[j].Position, &depthSpacePosition[j]);
					}

					//------------------------hand state left-------------------------------
					DrawHandState(depthSpacePosition[JointType_HandLeft], leftHandState);
					DrawHandState(depthSpacePosition[JointType_HandRight], rightHandState);

					//---------------------------body-------------------------------
					DrawBone(joints, depthSpacePosition, JointType_Head, JointType_Neck);
					DrawBone(joints, depthSpacePosition, JointType_Neck, JointType_SpineShoulder);
					DrawBone(joints, depthSpacePosition, JointType_SpineShoulder, JointType_SpineMid);
					DrawBone(joints, depthSpacePosition, JointType_SpineMid, JointType_SpineBase);
					DrawBone(joints, depthSpacePosition, JointType_SpineShoulder, JointType_ShoulderRight);
					DrawBone(joints, depthSpacePosition, JointType_SpineShoulder, JointType_ShoulderLeft);
					DrawBone(joints, depthSpacePosition, JointType_SpineBase, JointType_HipRight);
					DrawBone(joints, depthSpacePosition, JointType_SpineBase, JointType_HipLeft);

					// -----------------------Right Arm ------------------------------------
					DrawBone(joints, depthSpacePosition, JointType_ShoulderRight, JointType_ElbowRight);
					DrawBone(joints, depthSpacePosition, JointType_ElbowRight, JointType_WristRight);
					DrawBone(joints, depthSpacePosition, JointType_WristRight, JointType_HandRight);
					DrawBone(joints, depthSpacePosition, JointType_HandRight, JointType_HandTipRight);
					DrawBone(joints, depthSpacePosition, JointType_WristRight, JointType_ThumbRight);

					//----------------------------------- Left Arm--------------------------
					DrawBone(joints, depthSpacePosition, JointType_ShoulderLeft, JointType_ElbowLeft);
					DrawBone(joints, depthSpacePosition, JointType_ElbowLeft, JointType_WristLeft);
					DrawBone(joints, depthSpacePosition, JointType_WristLeft, JointType_HandLeft);
					DrawBone(joints, depthSpacePosition, JointType_HandLeft, JointType_HandTipLeft);
					DrawBone(joints, depthSpacePosition, JointType_WristLeft, JointType_ThumbLeft);

					// ----------------------------------Right Leg--------------------------------
					DrawBone(joints, depthSpacePosition, JointType_HipRight, JointType_KneeRight);
					DrawBone(joints, depthSpacePosition, JointType_KneeRight, JointType_AnkleRight);
					DrawBone(joints, depthSpacePosition, JointType_AnkleRight, JointType_FootRight);

					// -----------------------------------Left Leg---------------------------------
					DrawBone(joints, depthSpacePosition, JointType_HipLeft, JointType_KneeLeft);
					DrawBone(joints, depthSpacePosition, JointType_KneeLeft, JointType_AnkleLeft);
					DrawBone(joints, depthSpacePosition, JointType_AnkleLeft, JointType_FootLeft);

					delete[] depthSpacePosition;
					logger->debug("Drew skeleton for body {}", i);
				} else {
					logger->error("Failed to get joints for body {}", i);
				}
			}
		}
	}
	logger->debug("Finished processing. Found {} actually tracked bodies", trackedBodies);
}

void KinectDepthChecker::DrawBone(const Joint* pJoints, const DepthSpacePoint* depthSpacePosition, JointType joint0, JointType joint1) {
	TrackingState joint0State = pJoints[joint0].TrackingState;
	TrackingState joint1State = pJoints[joint1].TrackingState;

	// Log joint positions and states
	logger->debug("Drawing bone between joints {} and {}", joint0, joint1);
	logger->debug("Joint positions: ({}, {}) to ({}, {})",
		depthSpacePosition[joint0].X, depthSpacePosition[joint0].Y,
		depthSpacePosition[joint1].X, depthSpacePosition[joint1].Y);

	if ((joint0State == TrackingState_NotTracked) || (joint1State == TrackingState_NotTracked)) {
		logger->debug("Joint not tracked, skipping bone");
		return;
	}

	if ((joint0State == TrackingState_Inferred) && (joint1State == TrackingState_Inferred)) {
		logger->debug("Both joints inferred, skipping bone");
		return;
	}

	// Create points for line drawing
	cv::Point p1(static_cast<int>(depthSpacePosition[joint0].X),
				 static_cast<int>(depthSpacePosition[joint0].Y));
	cv::Point p2(static_cast<int>(depthSpacePosition[joint1].X),
				 static_cast<int>(depthSpacePosition[joint1].Y));

	// Check if points are within image bounds
	if (p1.x >= 0 && p1.x < cDepthWidth && p1.y >= 0 && p1.y < cDepthHeight &&
		p2.x >= 0 && p2.x < cDepthWidth && p2.y >= 0 && p2.y < cDepthHeight) {

		if ((joint0State == TrackingState_Tracked) && (joint1State == TrackingState_Tracked)) {
			cv::line(skeletonImg, p1, p2, cv::Scalar(255, 255, 0), 4);
			logger->debug("Drew tracked bone in yellow");
		} else {
			cv::line(skeletonImg, p1, p2, cv::Scalar(255, 0, 0), 4);
			logger->debug("Drew inferred bone in red");
		}
		} else {
			logger->debug("Points outside image bounds, skipping bone");
		}
}

void KinectDepthChecker::DrawHandState(const DepthSpacePoint depthSpacePosition, HandState handState) {
	cv::Scalar color;
	switch (handState) {
		case HandState_Open:
			color = cv::Scalar(255, 0, 0);
		break;
		case HandState_Closed:
			color = cv::Scalar(0, 255, 0);
		break;
		case HandState_Lasso:
			color = cv::Scalar(0, 0, 255);
		break;
		default:
			return;
	}

	circle(skeletonImg,
		cv::Point(depthSpacePosition.X, depthSpacePosition.Y),
		20, color, -1);
}
