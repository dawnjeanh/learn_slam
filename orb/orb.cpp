#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
int main(int argc, char **argv)
{

  if (argc < 3)
  {
    std::cout << "param err" << std::endl;
    return -1;
  }
  std::cout << cv::getBuildInformation() << std::endl;
  cv::Mat img1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
  cv::Mat img2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

  std::vector<cv::KeyPoint> keypoint1, keypoint2;
  orb->detect(img1, keypoint1);
  orb->detect(img2, keypoint2);

  cv::Mat description1, description2;
  orb->compute(img1, keypoint1, description1);
  orb->compute(img2, keypoint2, description2);

  std::vector<cv::DMatch> matches;
  matcher->match(description1, description2, matches);

  std::vector<cv::DMatch> good_matches;
  double min_dist = std::min_element(
                        matches.begin(), matches.end(),
                        [](cv::DMatch &m1, cv::DMatch &m2) {
                          return m1.distance < m2.distance;
                        })
                        ->distance;
  for (auto m : matches)
  {
    if (m.distance <= std::max(2 * min_dist, 30.0))
    {
      good_matches.push_back(m);
    }
  }

  cv::Mat out1, out2;
  cv::drawMatches(img1, keypoint1, img2, keypoint2, matches, out1);
  cv::drawMatches(img1, keypoint1, img2, keypoint2, good_matches, out2);
  cv::imshow("img1", out1);
  cv::imshow("img2", out2);
  cv::waitKey();

  return 0;
}
