#include "reader.h"

const string generate_string(const string& prefix, const int counter, const int number_of_digits)
{
	int check = counter == 1  ? 2 : counter;
	std::string dest = std::string(number_of_digits -  (log10(check) + 1), '0').append(to_string(counter) + ".jpg");
	return prefix + dest;
}
std::vector<int> unique(cv::Mat& input, bool sort = true)
{
	if (input.channels() > 1 || input.type() != 0)
	{
		std::cerr << "unique !!! Only works with CV_32F 1-channel Mat" << std::endl;
		return {};
	}

	std::vector<int> out;
	for (int y = 0; y < input.rows; ++y)
	{
		const uchar* row_ptr = input.ptr<uchar>(y);
		for (int x = 0; x < input.cols; ++x)
		{
			uchar value = row_ptr[x];

			if ( std::find(out.begin(), out.end(), value) == out.end() )
				out.push_back(value);
		}
	}

	if (sort)
		std::sort(out.begin(), out.end());

	for(auto & p : out)
		p = int(p);
	return out;
}
SimpleStruct Reader::get_frame(const int& step) {
	if (speeds.size() > counter + step)
	{
		Mat current = imread(generate_string(prefix_to_src_images, counter, 6));
		Mat next = imread(generate_string(prefix_to_src_images, counter + step, 6));
		Mat mask = imread(generate_string(prefix_to_segmentation_masks, counter, 6));
		cvtColor(mask, mask, cv::COLOR_RGB2GRAY);
		int dt = times[counter + step] - times[counter];
		float speed = speeds[counter];
		counter += step;
		return { current, next, mask, speed, dt};
	}
	else
		return { Mat(), Mat(), Mat(), 999, 0 };
}