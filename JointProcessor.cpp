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
    // Check confidence score if available
    float confidence = 0.0f;
    if (currentJoints) {
        auto confidenceIt = jointConfidences.find(jointId);
        if (confidenceIt != jointConfidences.end()) {
            confidence = confidenceIt->second;
        }
    }

    // Store confidence for later export
    lastProcessedConfidence[jointId] = confidence;

    // If confidence is very high (>0.9), trust the current position with minimal filtering
    if (confidence > 0.9f) {
        updateJointHistory(jointId, currentPos);
        return currentPos;
    }

    // Check if the current position is invalid (0,0,0)
    bool isCurrentValid = !(currentPos.x == 0 && currentPos.y == 0 && currentPos.z == 0);

    if (!isCurrentValid) {
        // If current position is invalid, but we have history or future frames, interpolate
        if (!jointHistory[jointId].empty() || (futurePositions && !futurePositions->empty())) {
            cv::Point3f interpolated = interpolateMissingJoint(jointId, futurePositions);
            // Only update history if we got a valid interpolation
            if (!(interpolated.x == 0 && interpolated.y == 0 && interpolated.z == 0)) {
                updateJointHistory(jointId, interpolated);
            }
            return interpolated;
        }

        // If no valid history or future data, try to estimate from neighboring joints
        if (currentJoints && !currentJoints->empty()) {
            cv::Point3f estimated = estimateJointFromNeighbors(jointId, *currentJoints);
            if (!(estimated.x == 0 && estimated.y == 0 && estimated.z == 0)) {
                updateJointHistory(jointId, estimated);
                return estimated;
            }
        }

        // No valid data available in either direction
        return cv::Point3f(0, 0, 0);
    } else {
        // Valid position, check if we have history
        if (!jointHistory[jointId].empty()) {
            // Get the most recent valid position from history
            cv::Point3f prevPos;
            int validHistoryCount = 0;

            // Find the most recent valid position in history
            for (auto it = jointHistory[jointId].rbegin(); it != jointHistory[jointId].rend(); ++it) {
                if (!(it->x == 0 && it->y == 0 && it->z == 0)) {
                    prevPos = *it;
                    validHistoryCount++;
                    break;
                }
            }

            // If no valid history found, just use current position
            if (validHistoryCount == 0) {
                updateJointHistory(jointId, currentPos);
                return currentPos;
            }

            // Calculate movement distance
            float distance = cv::norm(currentPos - prevPos);

            // Use enhanced movement validation based on joint type and time interval
            float maxAllowedMovement = getMaxJointMovement(jointId, timeInterval);
            bool validMovement = currentJoints ?
                isValidJointMovement(jointId, currentPos, prevPos, *currentJoints, timeInterval) :
                (distance <= maxAllowedMovement);

            // Apply adaptive filtering based on confidence and movement validity
            if (validMovement || confidence > 0.7f) {
                // Calculate filteredWeight - lower confidence means more filtering
                float filteredWeight = std::min(0.8f, std::max(0.05f, 0.9f - confidence));

                // If valid motion but low confidence, blend with history more heavily
                if (validMovement && confidence < 0.4f) {
                    filteredWeight = 0.6f;
                }

                // Blend current position with previous position
                cv::Point3f interpolated = (1.0f - filteredWeight) * currentPos + filteredWeight * prevPos;
                updateJointHistory(jointId, interpolated);
                return interpolated;
            }

            // Handle potentially invalid movement
            if (!validMovement) {
                // Get estimated position from neighbors if available
                cv::Point3f neighborEstimate(0, 0, 0);
                if (currentJoints && !currentJoints->empty()) {
                    neighborEstimate = estimateJointFromNeighbors(jointId, *currentJoints);
                }

                // Check future positions if available
                bool hasFutureData = false;
                cv::Point3f futureInfluence(0, 0, 0);
                float totalFutureWeight = 0.0f;

                if (futurePositions && !futurePositions->empty()) {
                    float futureWeight = 0.3f;

                    // Consider up to 3 future frames with decreasing weight
                    for (size_t i = 0; i < std::min(static_cast<size_t>(3), futurePositions->size()); i++) {
                        const cv::Point3f& futurePos = futurePositions->at(i);

                        // Skip invalid future positions
                        if (futurePos.x == 0 && futurePos.y == 0 && futurePos.z == 0) {
                            continue;
                        }

                        // Add weighted contribution from future frame
                        futureInfluence += futureWeight * futurePos;
                        totalFutureWeight += futureWeight;
                        futureWeight *= 0.5f; // Decrease weight for further frames
                        hasFutureData = true;
                    }
                }

                // Blend all available information
                if (hasFutureData || !(neighborEstimate.x == 0 && neighborEstimate.y == 0 && neighborEstimate.z == 0)) {
                    // Weight factors based on what data is available
                    float historyWeight = 0.6f - (confidence * 0.4f);
                    float currentWeight = std::max(0.05f, confidence * 0.3f);
                    float neighborWeight = 0.0f;

                    if (!(neighborEstimate.x == 0 && neighborEstimate.y == 0 && neighborEstimate.z == 0)) {
                        neighborWeight = 0.2f;
                        historyWeight *= 0.8f; // Reduce history weight if we have neighbor data
                    }

                    // Start with weighted history and current
                    cv::Point3f interpolated = historyWeight * prevPos + currentWeight * currentPos;

                    // Add neighbor data if available
                    if (neighborWeight > 0) {
                        interpolated += neighborWeight * neighborEstimate;
                    }

                    // Add future data if available
                    if (hasFutureData) {
                        interpolated += futureInfluence;
                        totalFutureWeight *= 0.8f; // Scale down future influence a bit
                    }

                    // Normalize weights
                    float totalWeight = historyWeight + currentWeight + neighborWeight + totalFutureWeight;
                    interpolated *= (1.0f / totalWeight);

                    updateJointHistory(jointId, interpolated);
                    return interpolated;
                }

                // Fallback to history-biased approach if no future/neighbor data
                float currentWeight = std::max(0.1f, confidence * 0.3f);
                cv::Point3f interpolated = (1.0f - currentWeight) * prevPos + currentWeight * currentPos;
                updateJointHistory(jointId, interpolated);
                return interpolated;
            }
        }

        // No history or valid movement, just use current position
        updateJointHistory(jointId, currentPos);
        return currentPos;
    }
}


cv::Point3f JointProcessor::interpolateMissingJoint(int jointId, const std::vector<cv::Point3f>* futurePositions) {
    // Check if we have any data to work with
    bool hasHistory = false;
    cv::Point3f lastValidHistoryPos(0, 0, 0);
    int validHistoryCount = 0;

    // Find valid history points
    if (!jointHistory[jointId].empty()) {
        for (auto it = jointHistory[jointId].rbegin(); it != jointHistory[jointId].rend(); ++it) {
            if (!(it->x == 0 && it->y == 0 && it->z == 0)) {
                if (validHistoryCount == 0) {
                    lastValidHistoryPos = *it;
                }
                validHistoryCount++;
            }
        }
        hasHistory = (validHistoryCount > 0);
    }

    // Check for valid future points
    bool hasFuture = false;
    int validFutureCount = 0;
    cv::Point3f firstValidFuturePos(0, 0, 0);

    if (futurePositions && !futurePositions->empty()) {
        for (const auto& pos : *futurePositions) {
            if (!(pos.x == 0 && pos.y == 0 && pos.z == 0)) {
                if (validFutureCount == 0) {
                    firstValidFuturePos = pos;
                }
                validFutureCount++;
            }
        }
        hasFuture = (validFutureCount > 0);
    }

    // If we have both history and future, simple linear interpolation might work best
    if (hasHistory && hasFuture) {
        // Weighted average, favoring the most recent valid points
        float historyWeight = 0.6f;
        float futureWeight = 0.4f;

        cv::Point3f interpolated = (historyWeight * lastValidHistoryPos) +
                                   (futureWeight * firstValidFuturePos);

        return interpolated;
    }

    // If we only have history or future, use weighted averaging with decay
    cv::Point3f result(0, 0, 0);
    float totalWeight = 0.0f;

    // Add historical data with exponential decay if available
    if (hasHistory) {
        float historyWeight = 1.0f;
        int historyPointsUsed = 0;

        for (auto it = jointHistory[jointId].rbegin(); it != jointHistory[jointId].rend(); ++it) {
            // Skip invalid points
            if (it->x == 0 && it->y == 0 && it->z == 0) {
                continue;
            }

            result += historyWeight * (*it);
            totalWeight += historyWeight;
            historyPointsUsed++;

            // Use exponential decay
            historyWeight *= 0.7f;

            // Limit how far back we look
            if (historyPointsUsed >= 5) break;
        }
    }

    // Add future data with exponential decay if available
    if (hasFuture) {
        float futureWeight = 1.0f;
        int futurePointsUsed = 0;

        for (const auto& futurePos : *futurePositions) {
            // Skip invalid points
            if (futurePos.x == 0 && futurePos.y == 0 && futurePos.z == 0) {
                continue;
            }

            result += futureWeight * futurePos;
            totalWeight += futureWeight;
            futurePointsUsed++;

            // Use exponential decay
            futureWeight *= 0.7f;

            // Limit how far forward we look
            if (futurePointsUsed >= 5) break;
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
    // Calculate distance moved
    float distance = cv::norm(currentPos - prevPos);

    // Get the maximum expected movement for this joint type
    float maxMove = getMaxJointMovement(jointId, timeInterval);

    // If movement is within basic constraints, accept it
    if (distance <= maxMove) {
        return true;
    }

    // If movement is extremely large, it's likely an error
    if (distance > 3.0f * maxMove) {
        return false;
    }

    // Check if connected joints are also moving significantly
    // This handles cases where rapid but coordinated movement is happening
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

    // Get list of connected joints
    auto connIt = connectedJoints.find(jointId);
    if (connIt == connectedJoints.end()) {
        // No connection info, use basic threshold
        return distance <= (1.2f * maxMove);
    }

    // Check movement of connected joints
    int connectedMovingJoints = 0;
    float avgConnectedMovement = 0.0f;
    float maxConnectedMovement = 0.0f;

    for (int connJointId : connIt->second) {
        auto jointIt = currentJoints.find(connJointId);
        if (jointIt != currentJoints.end()) {
            auto historyIt = jointHistory.find(connJointId);
            if (historyIt != jointHistory.end() && !historyIt->second.empty()) {
                // Find the last valid position in history
                cv::Point3f connectedPrev;
                bool foundValidPrev = false;

                for (auto it = historyIt->second.rbegin(); it != historyIt->second.rend(); ++it) {
                    if (!(it->x == 0 && it->y == 0 && it->z == 0)) {
                        connectedPrev = *it;
                        foundValidPrev = true;
                        break;
                    }
                }

                if (!foundValidPrev) continue;

                cv::Point3f connectedCurrent = jointIt->second;

                // Skip invalid positions
                if ((connectedCurrent.x == 0 && connectedCurrent.y == 0 && connectedCurrent.z == 0) ||
                    (connectedPrev.x == 0 && connectedPrev.y == 0 && connectedPrev.z == 0)) {
                    continue;
                }

                // Calculate movement
                float connectedDist = cv::norm(connectedCurrent - connectedPrev);
                float connectedMaxMove = getMaxJointMovement(connJointId, timeInterval);

                // Update stats
                avgConnectedMovement += connectedDist;
                maxConnectedMovement = std::max(maxConnectedMovement, connectedDist);

                // If this joint is also moving a lot, count it
                if (connectedDist > (0.5f * connectedMaxMove)) {
                    connectedMovingJoints++;
                }
            }
        }
    }

    // If multiple connected joints are also moving significantly, this is likely
    // valid coordinated movement
    if (connectedMovingJoints >= 2) {
        // Allow greater movement if connected joints are also moving
        return distance <= (2.0f * maxMove);
    }

    // Single joint moving a lot is suspicious unless it's a hand/foot
    // These need more freedom as they can move independently
    if (jointId == JointType_HandLeft || jointId == JointType_HandRight ||
        jointId == JointType_HandTipLeft || jointId == JointType_HandTipRight ||
        jointId == JointType_ThumbLeft || jointId == JointType_ThumbRight ||
        jointId == JointType_FootLeft || jointId == JointType_FootRight) {
        return distance <= (1.8f * maxMove);
    }

    // For other joints, be more conservative
    return distance <= (1.2f * maxMove);
}

void JointProcessor::reset() {
    jointHistory.clear();
    jointConfidences.clear();
    lastProcessedConfidence.clear();
}

void JointProcessor::storeJointConfidence(int jointId, float confidence) {
    jointConfidences[jointId] = confidence;
}

std::map<int, float> JointProcessor::getJointConfidences() const {
    return lastProcessedConfidence;
}

void JointProcessor::calculateStabilityMetrics(const std::vector<std::map<int, cv::Point3f>>& rawJoints,
                                             const std::vector<std::map<int, cv::Point3f>>& filteredJoints,
                                             std::map<int, float>& rawStability,
                                             std::map<int, float>& filteredStability) {
    // We'll use standard deviation of position as a stability metric
    std::map<int, std::vector<cv::Point3f>> rawJointTrajectories;
    std::map<int, std::vector<cv::Point3f>> filteredJointTrajectories;

    // First collect all valid positions for each joint
    for (const auto& frame : rawJoints) {
        for (const auto& joint : frame) {
            if (!(joint.second.x == 0 && joint.second.y == 0 && joint.second.z == 0)) {
                rawJointTrajectories[joint.first].push_back(joint.second);
            }
        }
    }

    for (const auto& frame : filteredJoints) {
        for (const auto& joint : frame) {
            if (!(joint.second.x == 0 && joint.second.y == 0 && joint.second.z == 0)) {
                filteredJointTrajectories[joint.first].push_back(joint.second);
            }
        }
    }

    // Now calculate standard deviation for each joint trajectory
    for (auto& trajectory : rawJointTrajectories) {
        int jointId = trajectory.first;
        const auto& positions = trajectory.second;

        if (positions.size() < 3) continue;  // Need at least a few samples

        // Calculate mean position
        cv::Point3f mean(0, 0, 0);
        for (const auto& pos : positions) {
            mean += pos;
        }
        mean *= (1.0f / positions.size());

        // Calculate variance
        float variance = 0;
        for (const auto& pos : positions) {
            cv::Point3f diff = pos - mean;
            variance += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        }
        variance /= positions.size();

        // Standard deviation (in mm for typical depth units)
        rawStability[jointId] = std::sqrt(variance) * 1000.0f;
    }

    // Same for filtered trajectories
    for (auto& trajectory : filteredJointTrajectories) {
        int jointId = trajectory.first;
        const auto& positions = trajectory.second;

        if (positions.size() < 3) continue;

        cv::Point3f mean(0, 0, 0);
        for (const auto& pos : positions) {
            mean += pos;
        }
        mean *= (1.0f / positions.size());

        float variance = 0;
        for (const auto& pos : positions) {
            cv::Point3f diff = pos - mean;
            variance += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        }
        variance /= positions.size();

        filteredStability[jointId] = std::sqrt(variance) * 1000.0f;
    }
}