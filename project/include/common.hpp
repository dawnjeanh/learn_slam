#ifndef __COMMON__
#define __COMMON__

#include <vector>
#include <memory>
#include <algorithm>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include "config.hpp"
#include "camera.hpp"
#include "frame.hpp"
#include "mappoint.hpp"
#include "map.hpp"
#include "visual_odometry.hpp"

#endif // !__COMMON__