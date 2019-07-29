#ifndef __POSE_ESTIMATION__
#define __POSE_ESTIMATION__

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

void pose_estimation_2d2d(
    const std::vector<cv::KeyPoint> &keypoint1,
    const std::vector<cv::KeyPoint> &keypoint2,
    const std::vector<cv::DMatch> &matches,
    cv::Mat &R, cv::Mat &t)
{
    // 内参
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    std::vector<cv::Point2d> points1, points2;
    for (auto m : matches)
    {
        points1.push_back(keypoint1[m.queryIdx].pt);
        points2.push_back(keypoint2[m.trainIdx].pt);
    }
    //计算基础矩阵
    cv::Mat F = cv::findFundamentalMat(points1, points2, CV_FM_8POINT);
    std::cout << "F = \n"
              << F << std::endl;
    //计算本质矩阵
    cv::Point2d principal_point(325.1, 249.7); // 光心
    int focal_length = 521;                    // 焦距
    cv::Mat E = cv::findEssentialMat(points1, points2, focal_length, principal_point, cv::RANSAC);
    std::cout << "E = \n"
              << F << std::endl;
    //计算旋转与平移
    cv::recoverPose(E, points1, points2, R, t, focal_length, principal_point);
}

void pose_estimation_3d2d(
    const std::vector<cv::Point3f> &points3d,
    const std::vector<cv::Point2f> &points2d,
    const cv::Mat &K, cv::Mat &R, cv::Mat &t)
{
    cv::Mat r;
    cv::solvePnP(points3d, points2d, K, cv::Mat(), r, t, false, cv::SOLVEPNP_EPNP);
    cv::Rodrigues(r, R);
}

void bundle_adjustment(
    const std::vector<cv::Point3f> &points3d,
    const std::vector<cv::Point2f> &points2d,
    const cv::Mat &K, cv::Mat &R, cv::Mat &t)
{
    using g2oBlock = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>; // pose 6, landmark 3
    g2oBlock::LinearSolverType *linearSolver = new g2o::LinearSolverCSparse<g2oBlock::PoseMatrixType>();
    g2oBlock *solver_ptr = new g2oBlock(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();
    Eigen::Matrix3d R_mat;
    R_mat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(R_mat, Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0))));
    optimizer.addVertex(pose);
    int id = 1;
    for (const auto p : points3d)
    {
        g2o::VertexSBAPointXYZ *point = new g2o::VertexSBAPointXYZ();
        point->setId(id);
        id++;
        point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
        point->setMarginalized(true);
        optimizer.addVertex(point);
    }

    // camera
    g2o::CameraParameters *camera = new g2o::CameraParameters(
        K.at<double>(0, 0),
        Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)),
        0);
    camera->setId(0);
    optimizer.addParameter(camera);

    // edges
    id = 1;
    for (const auto p : points2d)
    {
        g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(id);
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(id)));
        edge->setVertex(1, pose);
        edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
        edge->setParameterId(0, 0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        id++;
    }

    // run
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    Eigen::Matrix4d T = Eigen::Isometry3d(pose->estimate()).matrix();
    R = (cv::Mat_<double>(3, 3) << T(0, 0), T(0, 1), T(0, 2),
         T(1, 0), T(1, 1), T(1, 2),
         T(2, 0), T(2, 1), T(2, 2));
    t = (cv::Mat_<double>(3, 1) << T(0, 3), T(1, 3), T(2, 3));
}

void pose_estimation_3d3d(
    const std::vector<cv::Point3f> &points1,
    const std::vector<cv::Point3f> &points2,
    cv::Mat &R, cv::Mat &t)
{
    // center
    cv::Point3f p1(0, 0, 0), p2(0, 0, 0);
    int N = points1.size();
    for (int i = 0; i < N; ++i)
    {
        p1 += points1[i];
        p2 += points2[i];
    }
    p1 /= N;
    p2 /= N;
    // points move center
    std::vector<cv::Point3f> pts1(N), pts2(N);
    for (int i = 0; i < N; ++i)
    {
        pts1[i] = points1[i] - p1;
        pts2[i] = points2[i] - p2;
    }
    // q1 * q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; ++i)
    {
        W += Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z) * Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z).transpose();
    }
    std::cout << "W = " << W << std::endl;
    // SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R_ = U * (V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    R = (cv::Mat_<double>(3, 3) << R_(0, 0), R_(0, 1), R_(0, 2),
         R_(1, 0), R_(1, 1), R_(1, 2),
         R_(2, 0), R_(2, 1), R_(2, 2));
    R = R.t();
    t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
    t = -R * t;
}

#endif // !__POSE_ESTIMATION__