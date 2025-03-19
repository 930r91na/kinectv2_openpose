#ifndef JOINT_PROCESSOR_H
#define JOINT_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <map>
#include <deque>
#include <Kinect.h>

class JointProcessor {
private:
    std::map<int, std::deque<cv::Point3f>> jointHistory;
    const int historySize = 5;
    const float maxJointMovement = 0.3f;

public:
    JointProcessor();
    ~JointProcessor();

    cv::Point3f getFilteredJointPosition(int jointId, cv::Point3f currentPos, const std::vector<cv::Point3f>* futurePositions, const std::map<int, cv::Point3f>* currentJoints, float timeInterval);
    void updateJointHistory(int jointId, cv::Point3f position);
    cv::Point3f interpolateMissingJoint(int jointId, const std::vector<cv::Point3f>* futurePositions);
    cv::Point3f estimateJointFromNeighbors(int jointId, const std::map<int, cv::Point3f>& currentJoints);
    cv::Point3f refineJointWithDepth(int jointId, cv::Point2f joint2D, const cv::Mat& depthImage);
    float getMaxJointMovement(int jointId, float timeInterval = 1.0f) const;
    bool isValidJointMovement(int jointId, const cv::Point3f& currentPos,
                             const cv::Point3f& prevPos,
                             const std::map<int, cv::Point3f>& currentJoints,
                             float timeInterval = 1.0f) const;

    void reset();
};

#endif