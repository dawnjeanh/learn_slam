#ifndef __VISUAL_ODOMETRY__
#define __VISUAL_ODOMETRY__

#include "common.hpp"
#include "map.hpp"
#include "frame.hpp"
#include "config.hpp"

namespace myslam
{

class VisualOdometry
{
public:
    using Ptr = std::shared_ptr<VisualOdometry>;
    enum VOState
    {
        INITIALIZING = -1,
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
    Sophus::SE3d T_c_r_estimated_; // the estimated pose of current frame
    int num_inliers_;               // number of inlier features in icp
    int num_lost_;                 // number of lost times
    // parameters
    int num_of_features_; // number of features
    double scale_factor_; // scale in image pyramid
    int level_pyramid_; // number of pyramid level
    float match_ratio_; // ratio for selecting good matches
    int max_num_lost_; // max number of continuous lost times
    int min_inliers_; // minimum inliers

    double key_frame_min_rot; // minimum rotation of two key frames
    double key_frame_min_trans; // minimun translation of two key frames

    VisualOdometry() : state_(INITIALIZING), ref_(nullptr), curr_(nullptr), map_(new Map), num_lost_(0), num_inliers_(0)
    {
        num_of_features_ = Config::get<int>("num_of_features");
        scale_factor_ = Config::get<double>("scale_factor");
        level_pyramid_ = Config::get<int>("level_pyramid");
        match_ratio_ = Config::get<float>("match_ratio");
        max_num_lost_ = Config::get<int>("max_num_lost");
        min_inliers_ = Config::get<int>("min_inliers");
    }
    ~VisualOdometry(){}
    bool addFrame(Frame::Ptr frame) // add a new frame
    {
        switch(state_)
        {
        case INITIALIZING:
            state_ = OK;
            curr_ = ref_ = frame;
            map_->insertKeyFrame(frame);
            // extract feature from first frame
            extractKeyPoints();
            computeDescriptors();
            // compute the 3D position of feature in ref frame
            setRef3DPoints();
            break;
        case OK:
            curr_ = frame;
            extractKeyPoints();
            computeDescriptors();
            featureMatching();
            poseEstimationPnP();
            if (checkEstimationPose() == true)
            {
                curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;
                ref_ = curr_;
                setRef3DPoints();
                num_lost_ = 0;
                if (checkKeyFrame() == true)
                {
                    addKeyFrame();
                }
            }
            else
            {
                num_lost_++;
                if (num_lost_ > max_num_lost_)
                {
                    state_ = LOST;
                }
                return false;
            }
            break;
        case LOST:
            std::cout << "VO has lost" << std::endl;
            break;
        }
        return true;
    }
protected:
    // inner operation
    void extractKeyPoints();
    void computeDescriptors();
    void featureMatching();
    void poseEstimationPnP();
    void setRef3DPoints();
    void addKeyFrame();
    bool checkEstimationPose();
    bool checkKeyFrame();
};

} // namespace myslam

#endif // !__VISUAL_ODOMETRY__
