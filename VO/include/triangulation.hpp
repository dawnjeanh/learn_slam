#ifndef __TRIANGULATION__
#define __TRIANGULATION__

#include <vector>
#include <opencv2/opencv.hpp>

cv::Point2f pixel2cam(const cv::Point2f &p, const cv::Mat &K)
{
    return cv::Point2f(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void triangulation(
    const std::vector<cv::KeyPoint> &keypoint1,
    const std::vector<cv::KeyPoint> &keypoint2,
    const std::vector<cv::DMatch> &matches,
    const cv::Mat &R, const cv::Mat &t,
    std::vector<cv::Point3d> &points)
{
    cv::Mat T1 = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0,
                  0, 1, 0, 0,
                  0, 0, 1, 0);
    cv::Mat T2 = (cv::Mat_<double>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
                  R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                  R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1,
                 0, 521.0, 249.7,
                 0, 0, 1);
    // 像素坐标 -> 相机坐标
    std::vector<cv::Point2f> pts1, pts2;
    for (auto m : matches)
    {
        pts1.push_back(pixel2cam(keypoint1[m.queryIdx].pt, K));
        pts2.push_back(pixel2cam(keypoint2[m.trainIdx].pt, K));
    }
    //三角法
    cv::Mat pts4d;
    cv::triangulatePoints(T1, T2, pts1, pts2, pts4d);
    //转成非齐次坐标
    for (int i = 0; i < pts4d.cols; ++i)
    {
        cv::Mat x = pts4d.col(i);
        x /= x.at<float>(3, 0);
        cv::Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
        points.push_back(p);
    }
}

#endif // !__TRIANGULATION__