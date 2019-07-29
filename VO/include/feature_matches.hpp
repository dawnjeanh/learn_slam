#ifndef __FEATURE_MATCHES__
#define __FEATURE_MATCHES__

#include <vector>
#include <opencv2/opencv.hpp>

void find_feature_matches(
    const cv::Mat &img1, const cv::Mat &img2,
    std::vector<cv::KeyPoint> &keypoints1,
    std::vector<cv::KeyPoint> &keypoints2,
    std::vector<cv::DMatch> &matches)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->detect(img1, keypoints1);
    orb->detect(img2, keypoints2);

    cv::Mat description1, description2;
    orb->compute(img1, keypoints1, description1);
    orb->compute(img2, keypoints2, description2);

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<cv::DMatch> _matches;
    matcher->match(description1, description2, _matches);

    double min_dist = std::min_element(
                          _matches.begin(), _matches.end(),
                          [](cv::DMatch &m1, cv::DMatch &m2) {
                              return m1.distance < m2.distance;
                          })
                          ->distance;
    for (auto m : _matches)
    {
        if (m.distance <= std::max(2 * min_dist, 30.0))
        {
            matches.push_back(m);
        }
    }
}

#endif // !__FEATURE_MATCHES__