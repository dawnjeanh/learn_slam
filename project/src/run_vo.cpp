#include <fstream>
#include <opencv2/viz/vizcore.hpp>
#include "common.hpp"
#include "config.hpp"
#include "camera.hpp"
#include "visual_odometry.hpp"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "usage: run_vo parameter_file" << std::endl;
        return 1;
    }

    myslam::Config::setParameterFile(argv[1]);
    myslam::VisualOdometry::Ptr vo(new myslam::VisualOdometry);
    std::string dataset_dir = myslam::Config::get<std::string>("dataset_dir");
    std::cout << "dataset dir: " << dataset_dir << std::endl;

    std::ifstream fin(dataset_dir + "/associate_with_groundtruth.txt");
    if (!fin)
    {
        std::cout << "associate_with_groundtruth.txt is not exist" << std::endl;
        return 1;
    }

    std::vector<std::string> rgb_files, depth_files;
    std::vector<double> rgb_times, depth_times;
    std::vector<cv::Point3f> ground;
    while (!fin.eof())
    {
        std::string rgb_file, depth_file;
        double rgb_time, depth_time, ground_time, tx, ty, tz, qx, qy, qz, qw;
        fin >> rgb_time >> rgb_file >> depth_time >> depth_file >> ground_time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        rgb_times.push_back(rgb_time);
        depth_times.push_back(depth_time);
        rgb_files.push_back(dataset_dir + "/" + rgb_file);
        depth_files.push_back(dataset_dir + "/" + depth_file);
        ground.push_back(cv::Point3f(tx, ty, tz));
        if (fin.good() == false)
            break;
    }
    fin.close();

    myslam::Camera::Ptr camera(new myslam::Camera);
    // visualization
    cv::viz::Viz3d vis("Visual Odometry");
    cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
    cv::Point3d cam_pos(0, -2.0, -2.0), cam_focal_point(0, 0, 0), cam_y_dir(0, 1, 0);
    cv::Affine3d cam_pose = cv::viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
    vis.setViewerPose(cam_pose);
    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
    vis.showWidget("World", world_coor);
    vis.showWidget("Camera", camera_coor);

    std::vector<cv::Point3f> path_vo, path_ground;
    cv::Point3f start;
    path_vo.push_back(cv::Point3f(0, 0, 0));
    path_ground.push_back(cv::Point3f(0, 0, 0));

    for (int i = 0; i < rgb_files.size(); ++i)
    {
        cv::Mat color = cv::imread(rgb_files[i], CV_LOAD_IMAGE_COLOR);
        cv::Mat depth = cv::imread(depth_files[i], CV_LOAD_IMAGE_UNCHANGED);
        if ((color.data == nullptr) || (depth.data == nullptr))
            break;
        myslam::Frame::Ptr pFrame = myslam::Frame::creatFrame();
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        pFrame->depth_ = depth;
        pFrame->time_stamp_ = rgb_times[i];
        vo->addFrame(pFrame);
        if (vo->state_ == myslam::VisualOdometry::LOST)
            break;
        Sophus::SE3d Tcw = pFrame->T_c_w_.inverse();
        // show the map and the camera pose
        Eigen::Matrix3d R = Tcw.rotationMatrix();
        Eigen::Vector3d t = Tcw.translation();
        if (i > 0)
        {
            Eigen::Vector3d pt, pt_;
            pt << path_vo[-1].x, path_vo[-1].y, path_vo[-1].z;
            pt_ = R * pt + t;
            path_vo.push_back(cv::Point3f(pt_(0, 0), pt_(1, 0), pt_(2, 0)));
            path_ground.push_back(ground[i] - ground[0]);
            cv::viz::WPolyLine path(path_vo, cv::viz::Color::white());
            cv::viz::WPolyLine path_(path_ground, cv::viz::Color::red());
            vis.showWidget("path", path);
            vis.showWidget("path_", path_);
        }

        cv::Affine3d M(
            cv::Affine3d::Mat3(
                R(0, 0), R(0, 1), R(0, 2),
                R(1, 0), R(1, 1), R(1, 2),
                R(2, 0), R(2, 1), R(2, 2)),
            cv::Affine3d::Vec3(
                t(0, 0), t(1, 0), t(2, 0)));
        vis.setWidgetPose("Camera", M);
        vis.spinOnce(1, false);
        cv::imshow("image", color);
        cv::waitKey(30);
    }

    return 0;
}