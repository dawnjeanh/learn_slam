#include <iostream>
#include <fstream>
#include <string>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>


using pclPoint = pcl::PointXYZRGB;
using pclPointCloud = pcl::PointCloud<pclPoint>;


int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "param err" << std::endl;
        return -1;
    }
    std::vector<cv::Mat> colorImgs, depthImgs;
    std::vector<Eigen::Isometry3d> poses;

    std::fstream fin(argv[3]);
    for (int i = 0; i < 5; ++i)
    {
        std::string path1(argv[1]);
        path1 += "/" + std::to_string(i + 1) + ".png";
        colorImgs.push_back(cv::imread(path1));

        std::string path2(argv[2]);
        path2 += "/" + std::to_string(i + 1) + ".pgm";
        depthImgs.push_back(cv::imread(path2, 2));

        double data[7];
        for (auto &d : data)
            fin >> d;
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Vector3d t(data[0], data[1], data[2]);
        Eigen::Isometry3d T(q);
        T.pretranslate(t);
        poses.push_back(T);
    }
    fin.close();

    //相机内参
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0;

    pclPointCloud::Ptr pointCloud(new pclPointCloud);
    for (int i = 0; i < 5; ++i)
    {
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];
        for (int v = 0; v < color.rows; ++v)
        {
            for (int u = 0; u < color.cols; ++u)
            {
                unsigned short d = depth.at<unsigned short>(v, u);
                if (d == 0)
                    continue;
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWord = T * point;

                pclPoint p;
                p.x = point[0];
                p.y = point[1];
                p.z = point[2];
                p.b = color.at<cv::Vec3b>(u, v)[0];
                p.g = color.at<cv::Vec3b>(u, v)[1];
                p.r = color.at<cv::Vec3b>(u, v)[2];
                pointCloud->points.push_back(p);
            }
        }
    }
    pointCloud->is_dense = false;
    pcl::io::savePCDFileBinary("map.pcd", *pointCloud);
    return 0;
}