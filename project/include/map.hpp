#ifndef __MAP__
#define __MAP__

#include "common.hpp"

namespace myslam
{

class Map
{
public:
    using Ptr = std::shared_ptr<Map>;
    std::unordered_map<uint32_t, MapPoint::Ptr> map_points_; // all landmarks
    std::unordered_map<uint32_t, Frame::Ptr> key_frames_; // all key frames
    
    Map(){}
    
    void insertKeyFrame(Frame::Ptr frame)
    {
        key_frames_[frame->id_] = frame;
    }
    void insertMapPoint(MapPoint::Ptr map_point)
    {
        map_points_[map_point->id_] = map_point;
    }
};

} // namespace myslam


#endif // !__MAP__