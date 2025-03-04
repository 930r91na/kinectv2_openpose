#pragma once

#include <string>
#include <vector>
#include "OpenPoseCapture.h" // For Person3D struct

class SkeletonExporter {
public:
    // Single frame export
    static bool saveTo3DSkeletonFile(const std::vector<Person3D>& people, const std::string& outputPath);

    // Multi-frame support
    static bool createSkeletonFile(const std::string& outputPath);
    static bool appendFrameToSkeletonFile(const std::vector<Person3D>& people,
                                         const std::string& outputPath,
                                         uint64_t timestamp);
    static bool finalizeSkeletonFile(const std::string& outputPath);

    // Alternative formats
    static bool saveToCSV(const std::vector<Person3D>& people, const std::string& outputPath);
};