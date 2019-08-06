#ifndef __VISUAL_ODOMETRY__
#define __VISUAL_ODOMETRY__

#include "common.hpp"

namespace myslam
{

class VisualOdometry
{
public:
    using Ptr = std::shared_ptr<VisualOdometry>;
    enum VOState
    {
        INITIALIZING = -1,
        OK = 0,
        LOST
    };

    VOState state_;                           // current VO status
    Map::Ptr map_;                            // map with all frames and map points
    Frame::Ptr ref_, curr_;                   // reference frame and current frame
    cv::Ptr<cv::ORB> orb_;                    // orb detector and computer
    std::vector<cv::Point3f> pts_3d_ref_;     // 3d points in reference frame
    std::vector<cv::KeyPoint> keypoint_curr_; // keypoints in current frame
    cv::Mat descriptors_curr_;                // descriptor in current frame
    cv::Mat descriptors_ref_;                 // descriptor in reference frame
    std::vector<cv::DMatch> feature_matches_;
    Sophus::SE3d T_c_r_estimated_; // the estimated pose of current frame
    int num_inliers_;              // number of inlier features in icp
    int num_lost_;                 // number of lost times
    // parameters
    int num_of_features_; // number of features
    double scale_factor_; // scale in image pyramid
    int level_pyramid_;   // number of pyramid level
    float match_ratio_;   // ratio for selecting good matches
    int max_num_lost_;    // max number of continuous lost times
    int min_inliers_;     // minimum inliers

    double key_frame_min_rot;   // minimum rotation of two key frames
    double key_frame_min_trans; // minimun translation of two key frames

    VisualOdometry() : state_(INITIALIZING), ref_(nullptr), curr_(nullptr), map_(new Map), num_lost_(0), num_inliers_(0)
    {
        num_of_features_ = Config::get<int>("num_of_features");
        scale_factor_ = Config::get<double>("scale_factor");
        level_pyramid_ = Config::get<int>("level_pyramid");
        match_ratio_ = Config::get<float>("match_ratio");
        max_num_lost_ = Config::get<int>("max_num_lost");
        min_inliers_ = Config::get<int>("min_inliers");
        orb_ = cv::ORB::create(num_of_features_, scale_factor_, level_pyramid_);
    }
    ~VisualOdometry() {}
    bool addFrame(Frame::Ptr frame) // add a new frame
    {
        switch (state_)
        {
        case INITIALIZING:
            state_ = OK;
            curr_ = ref_ = frame;
            // extract feature from first frame
            extractKeyPoints();
            computeDescriptors();
            // add first frame as key-frame
            addKeyFrame();
            break;
        case OK:
            curr_ = frame;
            extractKeyPoints();
            computeDescriptors();
            featureMatching();
            poseEstimationPnP();
            if (checkEstimationPose() == true)
            {
                curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;
                ref_ = curr_;
                setRef3DPoints();
                num_lost_ = 0;
                if (checkKeyFrame() == true)
                {
                    addKeyFrame();
                }
            }
            else
            {
                num_lost_++;
                if (num_lost_ > max_num_lost_)
                {
                    state_ = LOST;
                }
                return false;
            }
            break;
        case LOST:
            std::cout << "VO has lost" << std::endl;
            break;
        }
        return true;
    }

protected:
    // inner operation
    inline void extractKeyPoints()
    {
        orb_->detect(curr_->color_, keypoint_curr_);
    }
    inline void computeDescriptors()
    {
        orb_->compute(curr_->color_, keypoint_curr_, descriptors_curr_);
    }
    void featureMatching()
    {
        // match ref and curr, use opencv's brute force match
        std::vector<cv::DMatch> matches;
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        matcher.match(descriptors_ref_, descriptors_curr_, matches);
        // select the bset matches
        float min_dist = std::min_element(
                             matches.begin(), matches.end(),
                             [](const cv::DMatch &m1, const cv::DMatch &m2) {
                                 return (m1.distance < m2.distance);
                             })
                             ->distance;
        feature_matches_.clear();
        for (auto &m : matches)
        {
            if (m.distance < std::max<float>(min_dist * match_ratio_, 30.0))
            {
                feature_matches_.push_back(m);
            }
        }
    }
    void poseEstimationPnP()
    {
        std::vector<cv::Point3f> pts3d;
        std::vector<cv::Point2f> pts2d;
        for (auto m : feature_matches_)
        {
            pts3d.push_back(pts_3d_ref_[m.queryIdx]);
            pts2d.push_back(keypoint_curr_[m.trainIdx].pt);
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << ref_->camera_->fx_, 0, ref_->camera_->cx_,
                     0, ref_->camera_->fy_, ref_->camera_->cy_,
                     0, 0, 1);
        cv::Mat rvec, tvec, inliers;
        cv::solvePnPRansac(pts3d, pts2d, K, cv::Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);
        num_inliers_ = inliers.rows;
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        Eigen::Matrix3d eR;
        eR << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
        T_c_r_estimated_ = Sophus::SE3d(
            eR, Sophus::Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)));
        bundleAdjustment(pts3d, pts2d, inliers);
    }
    void bundleAdjustment(std::vector<cv::Point3f> &pts3d, std::vector<cv::Point2f> &pts2d, cv::Mat &inliers)
    {
        using g2oBlock = g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>>;
        std::unique_ptr<g2oBlock::LinearSolverType> linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2oBlock::PoseMatrixType>>();
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2oBlock>(std::move(linearSolver)));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);
        // vertex
        g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();
        pose->setId(0);
        pose->setEstimate(g2o::SE3Quat(T_c_r_estimated_.rotationMatrix(), T_c_r_estimated_.translation()));
        optimizer.addVertex(pose);
        // edges
        for (int i = 0; i < inliers.rows; ++i)
        {
            int idx = inliers.at<int>(i, 0);
            EdgeProjectXYZ2UVPoseOnly *edge = new EdgeProjectXYZ2UVPoseOnly();
            edge->setId(i);
            edge->setVertex(0, pose);
            edge->camera_ = curr_->camera_.get();
            edge->point_ = Eigen::Vector3d(pts3d[idx].x, pts3d[idx].y, pts3d[idx].z);
            edge->setMeasurement(Eigen::Vector2d(pts2d[idx].x, pts2d[idx].y));
            edge->setInformation(Eigen::Matrix2d::Identity());
            optimizer.addEdge(edge);
        }
        // optimize
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        T_c_r_estimated_ = Sophus::SE3d(pose->estimate().rotation(), pose->estimate().translation());
    }
    void setRef3DPoints()
    {
        // select the features with depth measurements
        pts_3d_ref_.clear();
        descriptors_ref_ = cv::Mat();
        for (int i = 0; i < keypoint_curr_.size(); ++i)
        {
            double d = ref_->getDepth(keypoint_curr_[i]);
            if (d > 0)
            {
                Eigen::Vector3d p_cam = ref_->camera_->pixel2camera(
                    Eigen::Vector2d(keypoint_curr_[i].pt.x, keypoint_curr_[i].pt.y));
                pts_3d_ref_.push_back(cv::Point3f(p_cam(0, 0), p_cam(1, 0), p_cam(2, 0)));
                descriptors_ref_.push_back(descriptors_curr_.row(i));
            }
        }
    }
    void addKeyFrame()
    {
        if (map_->key_frames_.empty())
        {
            // first key-frame, add all 3d points into map
            for (int i = 0; i < keypoint_curr_.size(); ++i)
            {
                double d = curr_->getDepth(keypoint_curr_[i]);
                if (d < 0)
                    continue;
                Eigen::Vector3d p_world = ref_->camera_->pixel2world(
                    Eigen::Vector2d(keypoint_curr_[i].pt.x, keypoint_curr_[i].pt.y),
                    curr_->T_c_w_, d
                );
                Eigen::Vector3d n = p_world - ref_->getCamCenter();
                n.normalize();
                MapPoint::Ptr map_point = MapPoint::createMapPoint(
                    p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
                );
                map_->insertMapPoint(map_point);
            }
        }
        map_->insertKeyFrame(curr_);
        ref_ = curr_;
    }
    bool checkEstimationPose()
    {
        if (num_inliers_ < min_inliers_)
        {
            std::cout << "reject because inlier is too small" << std::endl;
            return false;
        }
        Sophus::Vector6d d = T_c_r_estimated_.log();
        if (d.norm() > 5.0)
        {
            std::cout << "reject bacause motion is too large" << std::endl;
            return false;
        }
        return true;
    }
    bool checkKeyFrame()
    {
        Sophus::Vector6d d = T_c_r_estimated_.log();
        Eigen::Vector3d trans = d.head(3);
        Eigen::Vector3d rot = d.tail(3);
        return ((rot.norm() > key_frame_min_rot) ||
                (trans.norm() > key_frame_min_trans));
    }
};

} // namespace myslam

#endif // !__VISUAL_ODOMETRY__
