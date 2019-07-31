#ifndef __DIRECT_METHODS__
#define __DIRECT_METHODS__

#include <vector>
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

class EdgeSE3ProjectDirect : public g2o::BaseUnaryEdge<1, double, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSE3ProjectDirect() {}
    EdgeSE3ProjectDirect(Eigen::Vector3d point, float fx, float fy, float cx, float cy, cv::Mat *img)
        : x_world_(point), fx_(fx), fy_(fy), cx_(cx), cy_(cy), img_(img) {}
    virtual void computeError()
    {
        const g2o::VertexSE3Expmap *v = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        Eigen::Vector3d x_local = v->estimate().map(x_world_);
        float x = x_local[0] * fx_ / x_local[2] + cx_;
        float y = x_local[1] * fy_ / x_local[2] + cy_;
        if ((x - 4 < 0) || (x + 4 > img_->cols) || (y - 4 < 0) || (y + 4 > img_->rows))
        {
            _error(0, 0) = 0.0;
            this->setLevel(1);
        }
        else
        {
            _error(0, 0) = getPixelValue(x, y) - _measurement;
        }
    }
    virtual void linearizeOplus()
    {
        if (level() == 1)
        {
            _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
            return;
        }
        g2o::VertexSE3Expmap *vtx = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        Eigen::Vector3d xyz_trans = vtx->estimate().map(x_world_);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1 / xyz_trans[2];
        double invz_2 = invz * invz;
        float u = x * fx_ * invz + cx_;
        float v = y * fy_ * invz + cy_;
        Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;
        jacobian_uv_ksai(0, 0) = -x * y * invz_2 * fx_;
        jacobian_uv_ksai(0, 1) = (1 + (x * x * invz_2)) * fx_;
        jacobian_uv_ksai(0, 2) = -y * invz * fx_;
        jacobian_uv_ksai(0, 3) = invz * fx_;
        jacobian_uv_ksai(0, 4) = 0;
        jacobian_uv_ksai(0, 5) = -x * invz_2 * fx_;
        jacobian_uv_ksai(1, 0) = -(1 + y * y * invz_2) * fy_;
        jacobian_uv_ksai(1, 1) = x * y * invz_2 * fy_;
        jacobian_uv_ksai(1, 2) = x * invz * fy_;
        jacobian_uv_ksai(1, 3) = 0;
        jacobian_uv_ksai(1, 4) = invz * fy_;
        jacobian_uv_ksai(1, 5) = -y * invz_2 * fy_;
        Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;
        jacobian_pixel_uv(0, 0) = (getPixelValue(u + 1, v) - getPixelValue(u - 1, v)) / 2;
        jacobian_pixel_uv(0, 1) = (getPixelValue(u, v + 1) - getPixelValue(u, v - 1)) / 2;
        _jacobianOplusXi = jacobian_pixel_uv * jacobian_uv_ksai;
    }
    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &out) const {}

protected:
    // get a gray scal value from reference image (bilinear interpolated)
    inline float getPixelValue(float x, float y)
    {
        int ix = int(x);
        int iy = int(y);
        float xx = x - floor(x);
        float yy = y - floor(y);
        return float(
            (1 - xx) * (1 - yy) * img_->at<unsigned char>(iy, ix) +
            xx * (1 - yy) * img_->at<unsigned char>(iy, ix + 1) +
            (1 - xx) * yy * img_->at<unsigned char>(iy + 1, ix) +
            xx * yy * img_->at<unsigned char>(iy + 1, ix + 1));
    }

private:
    Eigen::Vector3d x_world_;                 // 3D point in word frame
    float cx_ = 0, cy_ = 0, fx_ = 0, fy_ = 0; // camera intrinsics
    cv::Mat *img_ = nullptr;                  //reference image
};

void pose_estimation_direct(
    const std::vector<Eigen::Vector3d> pos_world,
    const std::vector<float> pos_gray,
    cv::Mat *img, cv::Mat &K,
    cv::Mat &R, cv::Mat &t)
{
    using g2oBlock = g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>>; // pose 6, gray 1
    std::unique_ptr<g2oBlock::LinearSolverType> linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2oBlock::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2oBlock>(std::move(linearSolver)));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);
    // vertex
    g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();
    pose->setEstimate(g2o::SE3Quat(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Identity()));
    pose->setId(0);
    optimizer.addVertex(pose);
    // edges
    Eigen::Matrix3d K_;
    cv::cv2eigen(K, K_);
    for (int i = 0; i < pos_world.size(); ++i)
    {
        EdgeSE3ProjectDirect *edge = new EdgeSE3ProjectDirect(
            pos_world[i],
            K_(0, 0), K_(1, 1), K_(0, 2), K_(1, 2), img);
        edge->setVertex(0, pose);
        edge->setMeasurement(pos_gray[i]);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        edge->setId(i + 1);
        optimizer.addEdge(edge);
    }
    optimizer.initializeOptimization();
    optimizer.optimize(30);
    Eigen::Matrix4d T = Eigen::Isometry3d(pose->estimate()).matrix();
    R = (cv::Mat_<double>(3, 3) << T(0, 0), T(0, 1), T(0, 2),
         T(1, 0), T(1, 1), T(1, 2),
         T(2, 0), T(2, 1), T(2, 2));
    t = (cv::Mat_<double>(3, 1) << T(0, 3), T(1, 3), T(2, 3));
}

#endif // !__DIRECT_METHODS__