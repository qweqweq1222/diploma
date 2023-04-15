#include "reader.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

SimpleStruct Reader::get_frame(const int& step) {
	if (speeds.size() > counter + step)
	{
		Mat current = imread((*src_images).path().string());
		advance(src_images, step);
		Mat next = imread((*src_images).path().string());
		Mat mask = imread((*masks).path().string());
		Mat rails = imread((*rails_masks).path().string());
		cvtColor(mask, mask, COLOR_BGR2GRAY);
		advance(masks, step);
		advance(rails_masks, step);
		int dt = times[counter + step] - times[counter];
		float speed = speeds[counter];
		counter += step;
		return { current, next, mask, rails , speed, dt};
	}
	else
		return { Mat(), Mat(), Mat(), Mat(), 999, 0 };
}
