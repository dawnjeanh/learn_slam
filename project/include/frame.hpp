#ifndef __FRAME__
#define __FRAME__

#include "common.hpp"

namespace myslam
{

class Frame
{
public:
    using Ptr = std::shared_ptr<Frame>;
    uint32_t id_;        // id for this frame
    double time_stamp_;  // when it is recorded
    Sophus::SE3d T_c_w_; // transform from world to camera
    Camera::Ptr camera_;
    cv::Mat color_, depth_;

    Frame(uint32_t id) : id_(id), time_stamp_(0), T_c_w_(Sophus::SE3d()), camera_(nullptr), color_(cv::Mat()), depth_(cv::Mat()) {}
    Frame(uint32_t id, double time_stamp, Sophus::SE3d &T_c_w, Camera::Ptr camera, cv::Mat &color, cv::Mat &depth) : id_(id), time_stamp_(time_stamp), T_c_w_(T_c_w), camera_(camera), color_(color), depth_(depth) {}
    // factory function
    static Ptr creatFrame()
    {
        static uint32_t factory_id = 0;
        return Ptr(new Frame(factory_id++));
    }
    // get the depth in depth map
    double getDepth(const cv::KeyPoint &kp)
    {
        int x = cvRound(kp.pt.x);
        int y = cvRound(kp.pt.y);
        int dx[5] = {0, -1, 0, 1, 0};
        int dy[5] = {0, 0, -1, 0, 1};
        for (int i = 0; i < 5; ++i)
        {
            uint16_t d = depth_.at<uint16_t>(y + dy[i], x + dx[i]);
            if (d > 0)
                return double(d) / camera_->depth_scale_;
        }
        return -1.0;
    }
    // get camera center
    Eigen::Vector3d getCamCenter() const
    {
        return T_c_w_.inverse().translation();
    }
    // check if a point is in this frame
    bool isInFrame(const Eigen::Vector3d &pt_world)
    {
        Eigen::Vector3d pt_cam = camera_->world2camera(pt_world, T_c_w_);
        if (pt_cam(2, 0) < 0)
            return false;
        Eigen::Vector2d pt_pixel = camera_->camera2pixel(pt_cam);
        return (pt_pixel(0, 0) > 0) && (pt_pixel(1, 0) > 0) && (pt_pixel(0, 0) < color_.cols) && (pt_pixel(1, 0) < color_.rows);
    }
};

} // namespace myslam

#endif // !__FRAME__