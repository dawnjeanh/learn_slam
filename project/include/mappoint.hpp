#ifndef __MAP_POINT__
#define __MAP_POINT__

#include "common.hpp"

namespace myslam
{

class MapPoint
{
public:
    using Ptr = std::shared_ptr<MapPoint>;
    uint32_t id_;             // ID
    Eigen::Vector3d pos_;     // position in world
    Eigen::Vector3d norm_;    // normal of viewing direction
    cv::Mat descriptor_;      // descriptor for matching
    uint32_t observed_times_; // being observed by feature matching algo
    uint32_t correct_times_;  // being an inliner in pose extimation

    MapPoint(uint32_t id) : id_(id), pos_(Eigen::Vector3d(0, 0, 0)), norm_(Eigen::Vector3d(0, 0, 0)), observed_times_(0), correct_times_(0) {}
    MapPoint(uint32_t id, Eigen::Vector3d &position, Eigen::Vector3d &norm) : id_(id), pos_(position), norm_(norm), observed_times_(0), correct_times_(0) {}
    // factory function
    static Ptr createMapPoint()
    {
        static long factory_id = 0;
        return Ptr(new MapPoint(factory_id++));
    }
};

} // namespace myslam

#endif // !__MAP_POINT__