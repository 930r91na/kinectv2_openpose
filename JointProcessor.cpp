#include "JointProcessor.h"
#include <algorithm>
#include <numeric>

JointProcessor::JointProcessor()
{
}

JointProcessor::~JointProcessor() = default;

cv::Point3f JointProcessor::getFilteredJointPosition(int jointId, cv::Point3f currentPos,
                                                   const std::vector<cv::Point3f>* futurePositions,
                                                   const std::map<int, cv::Point3f>* currentJoints,
                                                   float timeInterval) {
    // Check if the current position is invalid (0,0,0)
    if ((currentPos.x == 0 && currentPos.y == 0 && currentPos.z == 0)) {
        // If current position is invalid, but we have history or future frames, interpolate
        if (!jointHistory[jointId].empty() || (futurePositions && !futurePositions->empty())) {
            return interpolateMissingJoint(jointId, futurePositions);
        }

        // No valid data available in either direction
        return cv::Point3f(0, 0, 0);
    } else {
        // Valid position, check if it's a reasonable movement from previous position
        if (!jointHistory[jointId].empty()) {
            cv::Point3f prevPos = jointHistory[jointId].back();

            // Use enhanced movement validation if current joints are provided
            bool validMovement = currentJoints ?
                isValidJointMovement(jointId, currentPos, prevPos, *currentJoints, timeInterval) :
                (cv::norm(currentPos - prevPos) <= maxJointMovement * timeInterval);

            if (!validMovement) {
                // Check future positions if available
                if (futurePositions && !futurePositions->empty()) {
                    // Calculate future trajectory
                    float futureWeight = 0.3f;
                    cv::Point3f futureInfluence(0, 0, 0);
                    float totalWeight = 0.0f;

                    // Consider up to 3 future frames with decreasing weight
                    for (size_t i = 0; i < std::min(static_cast<size_t>(3), futurePositions->size()); i++) {
                        const cv::Point3f& futurePos = futurePositions->at(i);

                        // Skip invalid future positions
                        if (futurePos.x == 0 && futurePos.y == 0 && futurePos.z == 0) {
                            continue;
                        }

                        // Add weighted contribution from future frame
                        futureInfluence += futureWeight * futurePos;
                        totalWeight += futureWeight;
                        futureWeight *= 0.5f; // Decrease weight for further frames
                    }

                    // Blend with history and current position
                    if (totalWeight > 0) {
                        float historyWeight = 0.6f;
                        float currentWeight = 0.1f;

                        cv::Point3f interpolated = historyWeight * prevPos +
                                                 currentWeight * currentPos +
                                                 (futureInfluence / totalWeight);

                        // Normalize weights
                        float totalBlendWeight = historyWeight + currentWeight + totalWeight;
                        interpolated *= (1.0f / totalBlendWeight);

                        updateJointHistory(jointId, interpolated);
                        return interpolated;
                    }
                }

                // Fallback to history-only approach if no future data
                cv::Point3f interpolated = 0.7f * prevPos + 0.3f * currentPos;
                updateJointHistory(jointId, interpolated);
                return interpolated;
            }
        }

        // Update history with valid position
        updateJointHistory(jointId, currentPos);
        return currentPos;
    }
}

cv::Point3f JointProcessor::interpolateMissingJoint(int jointId, const std::vector<cv::Point3f>* futurePositions) {
    // Check if we have any data to work with
    bool hasHistory = !jointHistory[jointId].empty();
    bool hasFuture = (futurePositions && !futurePositions->empty());

    if (!hasHistory && !hasFuture) {
        return cv::Point3f(0, 0, 0);
    }

    // Initialize result
    cv::Point3f result(0, 0, 0);
    float totalWeight = 0.0f;

    // Add historical data with exponential decay
    if (hasHistory) {
        float historyWeight = 1.0f;

        for (auto it = jointHistory[jointId].rbegin(); it != jointHistory[jointId].rend(); ++it) {
            // Skip invalid points
            if (it->x == 0 && it->y == 0 && it->z == 0) {
                continue;
            }

            result += historyWeight * (*it);
            totalWeight += historyWeight;
            historyWeight *= 0.7f; // Decay factor for history
        }
    }

    // Add future data with exponential decay
    if (hasFuture) {
        float futureWeight = 1.0f;

        for (const auto& futurePos : * futurePositions) {
            // Skip invalid points
            if (futurePos.x == 0 && futurePos.y == 0 && futurePos.z == 0) {
                continue;
            }

            result += futureWeight * futurePos;
            totalWeight += futureWeight;
            futureWeight *= 0.7f; // Decay factor for future
        }
    }

    // Normalize the result
    if (totalWeight > 0) {
        result *= (1.0f / totalWeight);
    }

    return result;
}

void JointProcessor::updateJointHistory(int jointId, cv::Point3f position) {
    if (jointHistory[jointId].size() >= historySize) {
        jointHistory[jointId].pop_front();
    }
    jointHistory[jointId].push_back(position);
}

cv::Point3f JointProcessor::estimateJointFromNeighbors(int jointId, const std::map<int, cv::Point3f>& currentJoints) {
    // Define joint connections - which joints should be used to estimate others
    static const std::map<int, std::vector<std::pair<int, float>>> jointConnections = {
        // {targetJoint, {{sourceJoint1, relativeWeight1}, {sourceJoint2, relativeWeight2}}}
        {JointType_ElbowLeft, {{JointType_ShoulderLeft, 0.5f}, {JointType_WristLeft, 0.5f}}},
        {JointType_ElbowRight, {{JointType_ShoulderRight, 0.5f}, {JointType_WristRight, 0.5f}}},
        {JointType_KneeLeft, {{JointType_HipLeft, 0.5f}, {JointType_AnkleLeft, 0.5f}}},
        {JointType_KneeRight, {{JointType_HipRight, 0.5f}, {JointType_AnkleRight, 0.5f}}},
        {JointType_SpineMid, {{JointType_SpineShoulder, 0.5f}, {JointType_SpineBase, 0.5f}}},
        {JointType_Neck, {{JointType_Head, 0.3f}, {JointType_SpineShoulder, 0.7f}}},
        {JointType_WristLeft, {{JointType_ElbowLeft, 0.7f}, {JointType_HandLeft, 0.3f}}},
        {JointType_WristRight, {{JointType_ElbowRight, 0.7f}, {JointType_HandRight, 0.3f}}},
        {JointType_AnkleLeft, {{JointType_KneeLeft, 0.7f}, {JointType_FootLeft, 0.3f}}},
        {JointType_AnkleRight, {{JointType_KneeRight, 0.7f}, {JointType_FootRight, 0.3f}}}
    };
    
    // If there are no defined relationships for this joint, return zero
    auto it = jointConnections.find(jointId);
    if (it == jointConnections.end()) {
        return cv::Point3f(0, 0, 0);
    }
    
    cv::Point3f estimatedPosition(0, 0, 0);
    float totalWeight = 0.0f;
    
    // Calculate the weighted average of connected joints
    for (const auto& connection : it->second) {
        int connectedJointId = connection.first;
        float weight = connection.second;
        
        auto jointIt = currentJoints.find(connectedJointId);
        if (jointIt != currentJoints.end()) {
            const cv::Point3f& connectedPos = jointIt->second;
            
            // Only use if not (0,0,0)
            if (!(connectedPos.x == 0 && connectedPos.y == 0 && connectedPos.z == 0)) {
                estimatedPosition += weight * connectedPos;
                totalWeight += weight;
            }
        }
    }
    
    if (totalWeight > 0) {
        estimatedPosition *= (1.0f / totalWeight);
        return estimatedPosition;
    }
    
    return cv::Point3f(0, 0, 0);
}

cv::Point3f JointProcessor::refineJointWithDepth(int jointId, cv::Point2f joint2D, const cv::Mat& depthImage) {
    constexpr int patchSize = 7; // Size of search patch
    constexpr int halfPatch = patchSize / 2;
    
    // If joint is outside image bounds, return zero
    if (joint2D.x < halfPatch || joint2D.y < halfPatch || 
        joint2D.x >= depthImage.cols - halfPatch || 
        joint2D.y >= depthImage.rows - halfPatch) {
        return cv::Point3f(0, 0, 0);
    }
    
    // Extract depth patch around joint
    cv::Rect patchRect(joint2D.x - halfPatch, joint2D.y - halfPatch, patchSize, patchSize);
    cv::Mat depthPatch = depthImage(patchRect);
    
    // Find median depth value in patch (ignoring zeros)
    std::vector<float> depthValues;
    depthValues.reserve(patchSize * patchSize);
    
    for (int y = 0; y < depthPatch.rows; ++y) {
        for (int x = 0; x < depthPatch.cols; ++x) {
            float depth;
            if (depthImage.type() == CV_16UC1) {
                depth = static_cast<float>(depthPatch.at<uint16_t>(y, x));
            } else {
                depth = depthPatch.at<float>(y, x);
            }
            
            if (depth > 0) {
                depthValues.push_back(depth);
            }
        }
    }
    
    if (depthValues.empty()) {
        return cv::Point3f(0, 0, 0);
    }
    
    // Find median depth
    std::ranges::sort(depthValues);
    float medianDepth = depthValues[depthValues.size() / 2];

    // Return the 3D position
    return cv::Point3f(joint2D.x, joint2D.y, medianDepth);
}

float JointProcessor::getMaxJointMovement(int jointId, float timeInterval) const {
    // Base movement speeds for different joint types (meters per second)
    static const std::map<int, float> baseJointSpeeds = {
        {JointType_Head, 0.7f},
        {JointType_Neck, 0.5f},

        // Torso joints - moderate movement
        {JointType_SpineShoulder, 0.6f},
        {JointType_SpineMid, 0.6f},
        {JointType_SpineBase, 0.6f},

        // Arm joints - fast movement
        {JointType_ShoulderLeft, 0.8f},
        {JointType_ShoulderRight, 0.8f},
        {JointType_ElbowLeft, 1.0f},
        {JointType_ElbowRight, 1.0f},
        {JointType_WristLeft, 1.5f},
        {JointType_WristRight, 1.5f},
        {JointType_HandLeft, 1.8f},
        {JointType_HandRight, 1.8f},
        {JointType_HandTipLeft, 2.0f},
        {JointType_HandTipRight, 2.0f},
        {JointType_ThumbLeft, 1.8f},
        {JointType_ThumbRight, 1.8f},

        // Leg joints - moderate to fast movement
        {JointType_HipLeft, 0.7f},
        {JointType_HipRight, 0.7f},
        {JointType_KneeLeft, 1.2f},
        {JointType_KneeRight, 1.2f},
        {JointType_AnkleLeft, 1.5f},
        {JointType_AnkleRight, 1.5f},
        {JointType_FootLeft, 1.7f},
        {JointType_FootRight, 1.7f}
    };

    float baseSpeed = 1.0f; // Default if not found
    auto it = baseJointSpeeds.find(jointId);
    if (it != baseJointSpeeds.end()) {
        baseSpeed = it->second;
    }

    // Scale by time interval to get maximum movement
    return baseSpeed * timeInterval;
}

bool JointProcessor::isValidJointMovement(int jointId,
                                         const cv::Point3f& currentPos,
                                         const cv::Point3f& prevPos,
                                         const std::map<int, cv::Point3f>& currentJoints,
                                         float timeInterval) const {
    float distance = norm(currentPos - prevPos);

    float maxMove = getMaxJointMovement(jointId, timeInterval);

    if (distance <= maxMove) {
        return true;
    }

    static const std::map<int, std::vector<int>> connectedJoints = {
        {JointType_Head, {JointType_Neck}},
        {JointType_Neck, {JointType_Head, JointType_SpineShoulder}},
        {JointType_SpineShoulder, {JointType_Neck, JointType_SpineMid, JointType_ShoulderLeft, JointType_ShoulderRight}},
        {JointType_SpineMid, {JointType_SpineShoulder, JointType_SpineBase}},
        {JointType_SpineBase, {JointType_SpineMid, JointType_HipLeft, JointType_HipRight}},
        {JointType_ShoulderLeft, {JointType_SpineShoulder, JointType_ElbowLeft}},
        {JointType_ElbowLeft, {JointType_ShoulderLeft, JointType_WristLeft}},
        {JointType_WristLeft, {JointType_ElbowLeft, JointType_HandLeft}},
        {JointType_HandLeft, {JointType_WristLeft, JointType_HandTipLeft, JointType_ThumbLeft}},
        {JointType_ShoulderRight, {JointType_SpineShoulder, JointType_ElbowRight}},
        {JointType_ElbowRight, {JointType_ShoulderRight, JointType_WristRight}},
        {JointType_WristRight, {JointType_ElbowRight, JointType_HandRight}},
        {JointType_HandRight, {JointType_WristRight, JointType_HandTipRight, JointType_ThumbRight}},
        {JointType_HipLeft, {JointType_SpineBase, JointType_KneeLeft}},
        {JointType_KneeLeft, {JointType_HipLeft, JointType_AnkleLeft}},
        {JointType_AnkleLeft, {JointType_KneeLeft, JointType_FootLeft}},
        {JointType_HipRight, {JointType_SpineBase, JointType_KneeRight}},
        {JointType_KneeRight, {JointType_HipRight, JointType_AnkleRight}},
        {JointType_AnkleRight, {JointType_KneeRight, JointType_FootRight}}
    };

    auto connIt = connectedJoints.find(jointId);
    if (connIt != connectedJoints.end()) {
        int connectedMovingJoints = 0;
        float avgConnectedMovement = 0.0f;
        float avgConnectedThreshold = 0.0f;  // Track the average threshold for connected joints

        for (int connJointId : connIt->second) {
            auto jointIt = currentJoints.find(connJointId);
            if (jointIt != currentJoints.end()) {
                auto historyIt = jointHistory.find(connJointId);
                if (historyIt != jointHistory.end() && !historyIt->second.empty()) {
                    cv::Point3f connectedPrev = historyIt->second.back();
                    cv::Point3f connectedCurrent = jointIt->second;

                    if (!(connectedCurrent.x == 0 && connectedCurrent.y == 0 && connectedCurrent.z == 0) &&
                        !(connectedPrev.x == 0 && connectedPrev.y == 0 && connectedPrev.z == 0)) {

                        float connectedDist = cv::norm(connectedCurrent - connectedPrev);
                        avgConnectedMovement += connectedDist;

                        avgConnectedThreshold += getMaxJointMovement(connJointId, timeInterval);

                        connectedMovingJoints++;
                    }
                }
            }
        }

        if (connectedMovingJoints > 0) {
            avgConnectedMovement /= connectedMovingJoints;
            avgConnectedThreshold /= connectedMovingJoints;

            // Allow greater movement if connected joints are also moving
            // This handles cases like arm swinging or leg motion
            if (avgConnectedMovement > 0.5f * avgConnectedThreshold) {
                return distance <= 1.5f * maxMove;
            }
        }
    }

    return false;
}

void JointProcessor::reset() {
    jointHistory.clear();
}
