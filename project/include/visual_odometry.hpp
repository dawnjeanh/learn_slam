#ifndef __VISUAL_ODOMETRY__
#define __VISUAL_ODOMETRY__

#include "common.hpp"
#include "map.hpp"
#include "frame.hpp"

namespace myslam
{

class VisualOdometry
{
public:
    using Ptr = std::shared_ptr<VisualOdometry>;
    enum VOState
    {
        INITIALIZINF = -1,
        OK = 0,
        LOST
    };

    VOState state_;                           // current VO status
    Map::Ptr map_;                            // map with all frames and map points
    Frame::Ptr ref_, curr_;                   // reference frame and current frame
    cv::Ptr<cv::ORB> orb_;                    // orb detector and computer
    std::vector<cv::Point3f> pts_3d_ref_;     // 3d points in reference frame
    std::vector<cv::KeyPoint> keypoint_curr_; // keypoints in current frame
    cv::Mat descriptors_curr_;                // descriptor in current frame
    cv::Mat descriptors_ref_;                 // descriptor in reference frame
    std::vector<cv::DMatch> feature_matches_;
    Sophus::SE3d T_c_r_estimated_; //the estimated pose of current frame
    int num_inliers;               // number of inlier features in icp
    int num_lost_;                 // number of lost times
    // parameters

    VisualOdometry(/* args */);
    ~VisualOdometry();
};

} // namespace myslam

#endif // !__VISUAL_ODOMETRY__