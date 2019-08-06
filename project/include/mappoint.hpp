#ifndef __MAP_POINT__
#define __MAP_POINT__

#include "common.hpp"

namespace myslam
{

class MapPoint
{
public:
    using Ptr = std::shared_ptr<MapPoint>;
    uint32_t id_;                        // ID
    static uint32_t factory_id_;         // factory id
    bool good_;                          // whether a good point
    Eigen::Vector3d pos_;                // position in world
    Eigen::Vector3d norm_;               // normal of viewing direction
    cv::Mat descriptor_;                 // descriptor for matching
    std::list<Frame *> observed_frames_; // key-frames that can observe this point
    uint32_t visible_times_;             // being visible in current frame
    uint32_t matched_times_;             // being an inliner in pose extimation

    MapPoint(uint32_t id) : id_(id),
                            pos_(Eigen::Vector3d(0, 0, 0)),
                            norm_(Eigen::Vector3d(0, 0, 0)),
                            good_(true),
                            matched_times_(0),
                            visible_times_(0) {}
    MapPoint(uint32_t id,
             const Eigen::Vector3d &position,
             const Eigen::Vector3d &norm,
             Frame *frame, const cv::Mat &descriptor) : id_(id),
                                                        pos_(position),
                                                        norm_(norm),
                                                        good_(true),
                                                        matched_times_(1),
                                                        visible_times_(1),
                                                        descriptor_(descriptor)
    {
        observed_frames_.push_back(frame);
    }
    inline cv::Point3f getPositionCV() const
    {
        return cv::Point3f(pos_(0, 0), pos_(1, 0), pos_(2, 0));
    }
    // factory function
    static Ptr createMapPoint()
    {
        return Ptr(new MapPoint(factory_id_++));
    }
    static Ptr createMapPoint(const Eigen::Vector3d &pos_world,
                              const Eigen::Vector3d &norm,
                              const cv::Mat &descriptor, Frame *frame)
    {
        return Ptr(new MapPoint(factory_id_++, pos_world, norm, frame, descriptor));
    }
};

uint32_t MapPoint::factory_id_ = 0;

} // namespace myslam

#endif // !__MAP_POINT__