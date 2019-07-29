#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

int main(int argc, const char** argv) {
    Eigen::AngleAxisd v(M_PI / 4, Eigen::Vector3d(0, 0, 1));
    Eigen::Matrix3d m = v.toRotationMatrix();
    Eigen::Vector3d p(1, 0, 0);
    
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(v);

    Eigen::Quaterniond q = Eigen::Quaterniond(v);

    Eigen::Vector3d p_ = q * p;
    std::cout << p_.transpose() << std::endl;
    return 0;
}