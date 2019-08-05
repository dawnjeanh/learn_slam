#ifndef __CAMERA__
#define __CAMERA__

#include "common.hpp"
namespace myslam
{

class Camera
{
public:
    using Ptr = std::shared_ptr<Camera>;
    float fx_, fy_, cx_, cy_, depth_scale_;
    Camera()
    {
        fx_ = Config::get<float>("camera.fx");
        fy_ = Config::get<float>("camera.fy");
        cx_ = Config::get<float>("camera.cx");
        cy_ = Config::get<float>("camera.cy");
        depth_scale_ = Config::get<float>("camera.depth_scale");
    }
    Camera(float fx, float fy, float cx, float cy, float depth_scale) : fx_(fx), fy_(fy), cx_(cx), cy_(cy), depth_scale_(depth_scale) {}
    // coordinate transform: world, camera, pixel
    Eigen::Vector3d world2camera(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w)
    {
        return T_c_w * p_w;
    }
    Eigen::Vector3d camera2world(const Eigen::Vector3d &p_c, const Sophus::SE3d &T_c_w)
    {
        return T_c_w.inverse() * p_c;
    }
    Eigen::Vector2d camera2pixel(const Eigen::Vector3d &p_c)
    {
        return Eigen::Vector2d(
            fx_ * p_c(0, 0) / p_c(2, 0) + cx_,
            fy_ * p_c(1, 0) / p_c(2, 0) + cy_);
    }
    Eigen::Vector3d pixel2camera(const Eigen::Vector2d &p_p, double depth = 1)
    {
        return Eigen::Vector3d(
            (p_p(0, 0) - cx_) * depth / fx_,
            (p_p(1, 0) - cy_) * depth / fy_,
            depth);
    }
    Eigen::Vector3d pixel2world(const Eigen::Vector2d &p_p, const Sophus::SE3d &T_c_w, double depth = 1)
    {
        return camera2world(pixel2camera(p_p, depth), T_c_w);
    }
    Eigen::Vector2d world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w)
    {
        return camera2pixel(world2camera(p_w, T_c_w));
    }
};

} // namespace myslam

#endif //