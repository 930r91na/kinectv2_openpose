KinectV2 OpenPose Integrator
A system that combines Microsoft Kinect V2 depth sensing with OpenPose pose estimation for robust 3D skeleton tracking.
Overview
This project creates a bridge between the Microsoft Kinect V2 sensor and the OpenPose computer vision library. While Kinect provides real-time skeletal tracking with depth information, its accuracy and flexibility can be limited. OpenPose offers state-of-the-art 2D pose estimation but lacks depth information. By integrating these technologies, we get the best of both worlds: accurate pose detection with proper 3D coordinates.
Features

Two-phase workflow: Record frames first, process later
High-quality recording: Capture and store color and depth data
Flexible processing: Apply OpenPose with customizable settings
3D reconstruction: Map 2D poses to 3D using depth information
Multi-threaded processing: Efficient batch processing of recorded frames
Multiple export formats: JSON, CSV, and custom skeleton format
Configuration system: Easily adjust settings via config file or command line
Visualization tools: View detection results with depth-based coloring

Requirements

Windows 10
Microsoft Kinect V2 sensor and SDK
OpenPose (binary distribution)
OpenCV (for visualization and image processing)
Modern C++ compiler supporting C++17 or later
CMake for building

Dependencies

Kinect for Windows SDK 2.0
OpenPose
OpenCV
spdlog for logging
nlohmann/json for JSON handling

Installation

Install the Kinect for Windows SDK 2.0
Download OpenPose portable demo for Windows
Clone this repository
Build with CMake:
Copymkdir build
cd build
cmake ..
cmake --build . --config Release


Configuration
Create a config.ini file in the executable directory or use the command-line arguments to configure the application:
iniCopy# KinectOpenPose Configuration File
openpose_path = C:\path\to\OpenPose\bin\OpenPoseDemo.exe
net_resolution = 368
use_maximum_accuracy = false
keypoint_confidence_threshold = 40
process_every_n_frames = 15
recording_directory = recordings
output_directory = processed
Usage
Recording Mode
bashCopyKinectOpenPoseIntegrator.exe --record recording_name --interval 5 --compress
Controls during recording:

Press 'r' to Start/Stop recording
Press 's' to Save the current recording
Press 'q' to Quit

Processing Mode
bashCopyKinectOpenPoseIntegrator.exe --process recording_name --threads 8
Command Line Arguments

--record [name]: Recording mode (default)
--process [name]: Process an existing recording
--interval N: Process every Nth frame (default: 1)
--threads N: Use N threads for processing (default: 4)
--compress: Use video compression for color frames
--config file.ini: Use specific config file (default: config.ini)
--help: Display help information

Project Structure

KinectDepthChecker: Handles Kinect initialization and frame capture
FrameRecorder: Manages recording and storing of color and depth data
OpenPoseCapture: Interfaces with OpenPose and processes frames
ConfigManager: Handles loading and saving configuration
SkeletonExporter: Exports skeleton data in various formats

Output Data
Processed data is saved to the processed/[recording_name] directory and includes:

JSON files with 3D keypoint data
Visualization images showing detected skeletons
Optional skeleton and CSV exports

Acknowledgements

OpenPose by CMU Perceptual Computing Lab
Microsoft for the Kinect SDK
