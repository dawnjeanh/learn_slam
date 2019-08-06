#ifndef __COMMON__
#define __COMMON__

#include <vector>
#include <list>
#include <memory>
#include <algorithm>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include "config.hpp"
#include "camera.hpp"
#include "frame.hpp"
#include "mappoint.hpp"
#include "map.hpp"
#include "g2o_types.hpp"
#include "visual_odometry.hpp"

#endif // !__COMMON__