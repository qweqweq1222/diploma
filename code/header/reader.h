#pragma once

#include <opencv2/core.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Camera {
  cv::Mat K, R0, t0;
};

struct SimpleStruct
{
  cv::Mat current, next, mask, rails;
	float speed;
	int dt;
};
struct KeyPointMatches
{
  std::vector<cv::DMatch> matches;
  std::vector<cv::KeyPoint> kp1, kp2;

	KeyPointMatches(std::vector<cv::DMatch> matches_, std::vector<cv::KeyPoint> kp1_,
		std::vector<cv::KeyPoint> kp2_) :matches(matches_), kp1(kp1_), kp2(kp2_) {};
	~KeyPointMatches() = default;
};
class Reader {
public:
	Reader(fs::directory_iterator& src, fs::directory_iterator& masks, const std::string& time, const std::string& speed) :
		src_images(src), masks(masks)
	{
    std::fstream times_file(time, std::ios_base::in);
    std::fstream speed_file(speed, std::ios_base::in);
		int t;
		float v;
		while (times_file >> t)
			times.emplace_back(t);
		while (speed_file >> v)
			speeds.emplace_back(v);
	}

	Reader(fs::directory_iterator& src, fs::directory_iterator& masks, fs::directory_iterator& rails_masks, const std::string& time, const std::string& speed) :
		src_images(src), masks(masks), rails_masks(rails_masks)
	{
    std::fstream times_file(time, std::ios_base::in);
    std::fstream speed_file(speed, std::ios_base::in);
		int t;
		float v;
		while (times_file >> t)
			times.emplace_back(t);
		while (speed_file >> v)
			speeds.emplace_back(v);
	}

	SimpleStruct get_frame(const int& step);
	~Reader() = default;
	fs::directory_iterator src_images;
	fs::directory_iterator masks;
	fs::directory_iterator rails_masks;
  std::vector<int> times;
  std::vector<float> speeds;
	int counter = 0;
};
