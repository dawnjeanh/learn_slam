#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "pose_estimation.hpp"
#include "feature_matches.hpp"
#include "triangulation.hpp"
#include "direct_methods.hpp"

int main(int argc, char const *argv[])
{
    if (argc < 5)
    {
        std::cout << "param err" << std::endl;
        return -1;
    }
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1,
                 0, 521.0, 249.7,
                 0, 0, 1);
    cv::Mat img1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat img1depth = cv::imread(argv[2], CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat img2 = cv::imread(argv[3], CV_LOAD_IMAGE_COLOR);
    cv::Mat img2depth = cv::imread(argv[4], CV_LOAD_IMAGE_UNCHANGED);
    // 2D-2D
    std::cout << "========== 2D-2D ==========" << std::endl;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(img1, img2, keypoints1, keypoints2, matches);
    std::cout << "match " << matches.size() << " points" << std::endl;
    cv::Mat R, t, v;
    pose_estimation_2d2d(keypoints1, keypoints2, matches, R, t);
    cv::Rodrigues(R, v);
    std::cout << "R = " << std::endl
              << R << std::endl
              << "v = " << std::endl
              << cv::norm(v, cv::NORM_L2) << std::endl
              << "t = " << std::endl
              << t << std::endl;

    std::vector<cv::Point3d> points;
    triangulation(keypoints1, keypoints2, matches, R, t, points);
    for (int i = 0; i < matches.size(); ++i)
    {
        cv::Point2d pt_cam1 = pixel2cam(keypoints1[matches[i].queryIdx].pt, K);
        cv::Point2d pt_cma1_3d(points[i].x / points[i].z, points[i].y / points[i].z);
        // std::cout << "point 2D: " << pt_cam1 << " --- 3D: " << pt_cma1_3d << " " << points[i].z << std::endl;
    }

    // 3D-2D
    // 由深度计算图1的三维坐标
    std::cout << "========== 3D-2D ==========" << std::endl;
    std::vector<cv::Point2f> pts2d;
    std::vector<cv::Point3f> pts3d;
    for (auto m : matches)
    {
        int u = int(keypoints1[m.queryIdx].pt.y);
        int v = int(keypoints1[m.queryIdx].pt.x);
        unsigned short d = img1depth.at<unsigned short>(u, v);
        if (d == 0)
            continue;
        float dd = d / 1000.0;
        cv::Point2f pt1 = pixel2cam(keypoints1[m.queryIdx].pt, K);
        pts3d.push_back(cv::Point3f(pt1.x * dd, pt1.y * dd, dd));
        pts2d.push_back(cv::Point2f(keypoints2[m.trainIdx].pt));
    }
    cv::Mat R2, t2, v2;
    pose_estimation_3d2d(pts3d, pts2d, K, R2, t2);
    cv::Rodrigues(R2, v2);
    std::cout << "R = " << std::endl
              << R2 << std::endl
              << "v = " << std::endl
              << cv::norm(v2, cv::NORM_L2) << std::endl
              << "t = " << std::endl
              << t2 << std::endl;
    bundle_adjustment(pts3d, pts2d, K, R2, t2);
    cv::Rodrigues(R2, v2);
    std::cout << "R = " << std::endl
              << R2 << std::endl
              << "v = " << std::endl
              << cv::norm(v2, cv::NORM_L2) << std::endl
              << "t = " << std::endl
              << t2 << std::endl;

    // 3D-3D
    // 由深度计算图1的三维坐标
    std::cout << "========== 3D-3D ==========" << std::endl;
    std::vector<cv::Point3f> pts1;
    std::vector<cv::Point3f> pts2;
    for (auto m : matches)
    {
        int u = int(keypoints1[m.queryIdx].pt.y);
        int v = int(keypoints1[m.queryIdx].pt.x);
        unsigned short d = img1depth.at<unsigned short>(u, v);
        if (d == 0)
            continue;
        float d1 = d / 1000.0;
        cv::Point2f pt1 = pixel2cam(keypoints1[m.queryIdx].pt, K);

        u = int(keypoints2[m.trainIdx].pt.y);
        v = int(keypoints2[m.trainIdx].pt.x);
        d = img2depth.at<unsigned short>(u, v);
        if (d == 0)
            continue;
        float d2 = d / 1000.0;
        cv::Point2f pt2 = pixel2cam(keypoints2[m.trainIdx].pt, K);

        pts1.push_back(cv::Point3f(pt1.x * d1, pt1.y * d1, d1));
        pts2.push_back(cv::Point3f(pt2.x * d2, pt2.y * d2, d2));
    }
    std::cout << "find point pair " << pts1.size() << std::endl;
    cv::Mat R3, t3, v3;
    pose_estimation_3d3d(pts1, pts2, R3, t3);
    cv::Rodrigues(R3, v3);
    std::cout << "R = " << std::endl
              << R3 << std::endl
              << "v = " << std::endl
              << cv::norm(v3, cv::NORM_L2) << std::endl
              << "t = " << std::endl
              << t3 << std::endl;
    // direct method
    std::cout << "========== direct method ==========" << std::endl;
    // extract img1 FAST feature points
    std::vector<cv::KeyPoint> img1_fast_points;
    cv::Ptr<cv::FastFeatureDetector> fast_detector = cv::FastFeatureDetector::create();
    fast_detector->detect(img1, img1_fast_points);
    cv::Mat img1_gray;
    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    std::vector<Eigen::Vector3d> pos_world;
    std::vector<float> pos_gray;
    for (auto p : img1_fast_points)
    {
        if ((p.pt.x < 20) || (p.pt.x > img1.cols - 20) ||
            (p.pt.y < 20) || (p.pt.y > img1.cols - 20))
            continue;
        unsigned short d = img1depth.at<unsigned short>(cvRound(p.pt.y), cvRound(p.pt.x));
        if (d == 0)
            continue;
        float d1 = d / 1000.0;
        cv::Point2f pt1_cam = pixel2cam(p.pt, K);
        pos_world.push_back(Eigen::Vector3d(pt1_cam.x * d1, pt1_cam.y * d1, d1));
        pos_gray.push_back(float(img1_gray.at<u_int8_t>(cvRound(p.pt.y), cvRound(p.pt.x))));
    }
    cv::Mat img2_gray;
    cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);
    cv::Mat R4, t4, v4;
    pose_estimation_direct(pos_world, pos_gray, &img2_gray, K, R4, t4);
    cv::Rodrigues(R4, v4);
    std::cout << "R = " << std::endl
              << R4 << std::endl
              << "v = " << std::endl
              << cv::norm(v4, cv::NORM_L2) << std::endl
              << "t = " << std::endl
              << t4 << std::endl;
    return 0;
}
