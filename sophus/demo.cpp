#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

namespace Eigen
{
    using Vector6d = Matrix<double, 6, 1>;
}

int main(int argc, const char** argv) {
    //SO3
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
    Sophus::SO3d SO3(R);
    Eigen::Vector3d so3 = SO3.log();
    Eigen::Matrix3d so3_hat = Sophus::SO3d::hat(so3);
    Eigen::Vector3d so3_ = Sophus::SO3d::vee(so3_hat);

    //扰动模型
    Eigen::Vector3d update_so3(1e-4, 0, 0);
    Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3;

    //SE3
    Eigen::Vector3d t(1, 0, 0);
    Sophus::SE3d SE3(R, t);
    Eigen::Vector6d se3 = SE3.log();
    Eigen::Matrix4d se3_hat = Sophus::SE3d::hat(se3);
    Eigen::Vector6d se3_ = Sophus::SE3d::vee(se3_hat);

    //扰动模型
    Eigen::Vector6d update_se3;
    update_se3.setZero();
    update_se3(0, 0) = 1e-4;
    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3;
    std::cout << SE3_updated.log() << std::endl;
    return 0;
}