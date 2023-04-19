#pragma once
#include "reader.h"
#include "optim_struct.h"
# define M_PI  3.14159265358979323846

Mat reconstruct_from_v6(double* alpha_trans) // функция принимает на вход 3 угла и 3 координаты вектора смещения и возвращает матрицу 4x4 [R|t]
{
	Mat answer = Mat::eye(4, 4, CV_32F);
	double a = alpha_trans[0];
	double b = alpha_trans[1];
	double g = alpha_trans[2];
	answer.at<float>(0, 0) = cos(b) * cos(g);
	answer.at<float>(0, 1) = -sin(g) * cos(b);
	answer.at<float>(0, 2) = sin(b);
	answer.at<float>(1, 0) = sin(a) * sin(b) * cos(g) + sin(g) * cos(a);
	answer.at<float>(1, 1) = -sin(a) * sin(b) * sin(g) + cos(g) * cos(a);
	answer.at<float>(1, 2) = -sin(a) * cos(b);
	answer.at<float>(2, 0) = sin(a) * sin(g) - sin(b) * cos(a) * cos(g);
	answer.at<float>(2, 1) = sin(a) * cos(g) + sin(b) * sin(g) * cos(a);
	answer.at<float>(2, 2) = cos(a) * cos(b);
	answer.at<float>(0, 3) = alpha_trans[3];
	answer.at<float>(1, 3) = alpha_trans[4];
	answer.at<float>(2, 3) = alpha_trans[5];
	return answer;
}
Mat general_estimate(Point pt, Mat& R0, Mat& t0, Mat& K, Mat& plane) // по точке на матрице камеры (зная, что она лежит в глобальной СК в плоскости plane) оцениваем ее координаты в глобальной СК
{
	Mat Rt = Mat::eye(Size(4, 3), CV_32FC1);
	R0.copyTo(Rt(Rect(0, 0, 3, 3)));
	t0.copyTo(Rt(Rect(3, 0, 1, 3)));
	Mat P = K * Rt;
	Mat P_abcd = Mat::eye(Size(4, 4), CV_32FC1);
	P.copyTo(P_abcd(Rect(0, 0, 4, 3)));
	plane.copyTo(P_abcd(Rect(0, 3, 4, 1)));
	Mat answer(Size(1, 4), CV_32FC1);
	answer.at<float>(0, 0) = pt.x;
	answer.at<float>(1, 0) = pt.y;
	answer.at<float>(2, 0) = 1;
	answer.at<float>(3, 0) = 0;
	Mat pt3d = P_abcd.inv() * answer;
	pt3d /= pt3d.at<float>(3);
	return pt3d;
}
static cv::Mat1b normalize1b(const cv::Mat& src) // преобразует одноканальную картинку в 1-байтную беззнаковую, растягивая от минимума до максимума значения
{
	double minv(0), maxv(0);  minMaxLoc(src, &minv, &maxv);
	double denom = max(1., maxv - minv);
	cv::Mat1b res;
	src.convertTo(res, CV_8UC1, 255.0 / denom, minv / denom); /// хмм max-minv+1 ?
	return res;
}
void GetMaxim(Mat1b src, double& maxVal, Point& first, Point& second) // улучшенная версия без поиска руками
{
	double minVal = 0, maxval = 0;
	Point min;
	minMaxLoc(src, &minVal, &maxVal, &min,&first);
	//зануляем в целой окрестности (Прямоугольник размеров 25 x 25)
	for(int dy = -25; dy < 25; ++dy)
		for(int dx = -25; dx < 25; ++dx)
			if(first.y + dy > 0 && first.y + dy < src.rows && first.x + dx > 0 && first.x + dx < src.cols)
				src.at<uchar>(first.y + dy, first.x + dx) = 0;
	minMaxLoc(src, &minVal, &maxval, &min,&second);
}
vector<double> get_angles_and_vec(const Mat Rt) // Принимает на вход матрицу 4 x 4 [R|t] - возвращает 3 угла и 3 координаты вектора смещения
{
	double alpha, beta, gamma;
	if (abs(Rt.at<float>(0, 2)) < 1)
		beta = asin(Rt.at<float>(0, 2));
	else if (Rt.at<float>(0, 2) == 1)
		beta = M_PI / 2;
	else if (Rt.at<float>(0, 2) == -1)
		beta = -M_PI / 2;

	if (abs(Rt.at<float>(2, 2) / cos(beta)) < 1)
		alpha = acos(Rt.at<float>(2, 2) / cos(beta));
	else if (Rt.at<float>(2, 2) / cos(beta) == 1)
		beta = 0;
	else if (Rt.at<float>(2, 2) / cos(beta) == -1)
		beta = M_PI;

	if (abs(Rt.at<float>(0, 0) / cos(beta)) < 1)
		gamma = acos(Rt.at<float>(0, 0) / cos(beta));
	else if (Rt.at<float>(0, 0) / cos(beta) == 1)
		beta = 0;
	else if (Rt.at<float>(0, 0) / cos(beta) == -1)
		beta = M_PI;
	return { alpha, beta, gamma, Rt.at<float>(0,3), Rt.at<float>(1,3), Rt.at<float>(2,3) };
}
KeyPointMatches align_images(cv::Mat& current, cv::Mat& next, const int max_features = 5000) { // функция для детектирования особых точек

	cv::Mat im1Gray, im2Gray, descriptors1, descriptors2;
	resize(next, next, current.size());
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	std::vector< std::vector<cv::DMatch> > knn_matches;
	cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
	detector->detectAndCompute(current, cv::noArray(), keypoints1, descriptors1);
	detector->detectAndCompute(next, cv::noArray(), keypoints2, descriptors2);
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
	const float ratio_thresh = 0.5f;
	std::vector<cv::DMatch> good_matches;
	sort(knn_matches.begin(), knn_matches.end());
	double median = knn_matches[knn_matches.size() / 2].data()->distance;
	for (int i = 0; i < knn_matches.size(); ++i)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance && knn_matches[i][0].distance < median)
		{
			double dx = keypoints1[knn_matches[i][0].queryIdx].pt.x - keypoints2[knn_matches[i][0].trainIdx].pt.x;
			double dy = keypoints1[knn_matches[i][0].queryIdx].pt.y - keypoints2[knn_matches[i][0].trainIdx].pt.y;
			if (sqrt(dx * dx + dy * dy) < 100)
				good_matches.emplace_back(knn_matches[i][0]);
		}
	}
	if (good_matches.size() > max_features)
	{
		std::sort(good_matches.begin(), good_matches.end());
		good_matches.erase(good_matches.begin(), good_matches.begin() + max_features);
	}

	return KeyPointMatches(good_matches, keypoints1, keypoints2);
}

pair<Vec4i, Vec4i> get_rails(Mat& frame, Camera& camera)
{
	float scale = 0.75;
	const int height = frame.rows;
	const int width = frame.cols;
	int scale_view = scale * height;

	Mat roid = frame(Range(int(scale_view), height),cv::Range(0,height));
	cvtColor(roid, roid, COLOR_BGR2GRAY);
	ximgproc::FastHoughTransform(roid, roid, CV_32FC1, cv::ximgproc::ARO_CTR_VER,
		cv::ximgproc::FHT_AVE, cv::ximgproc::HDO_RAW);
	GaussianBlur(roid, roid, cv::Size(7, 7), 0, 0, cv::BORDER_DEFAULT);
	Mat1b res = normalize1b(roid);

	Mat1b res__ = res.clone(); // копия для отрисовки
	double maxVal = 0;
	Point first,second;
	GetMaxim(res, maxVal, first, second);
	auto ln = cv::ximgproc::HoughPoint2Line(Point(first), roid, cv::ximgproc::ARO_CTR_VER,cv::ximgproc::HDO_RAW);
	auto bn = cv::ximgproc::HoughPoint2Line(Point(second), roid, cv::ximgproc::ARO_CTR_VER,cv::ximgproc::HDO_RAW);

	ln[3] += scale_view;
	ln[1] += scale_view;
	bn[3] += scale_view;
	bn[1] += scale_view;

	return {{ln[0],ln[1],ln[2],ln[3]}, {bn[0],bn[1], bn[2],bn[3]}};
}
vector<bool> filter_rails(pair<Vec4i, Vec4i>& rails, Camera& camera, Mat& plane)
{

	Mat pt1 = general_estimate(Point(rails.first[0],rails.first[1]), camera.R0, camera.t0, camera.K, plane);
	Mat pt2 = general_estimate(Point(rails.first[2],rails.first[3]), camera.R0, camera.t0, camera.K, plane);
	Mat pt3 = general_estimate(Point(rails.second[0],rails.second[1]), camera.R0, camera.t0, camera.K, plane);
	Mat pt4 = general_estimate(Point(rails.second[2],rails.second[3]), camera.R0, camera.t0, camera.K, plane);

	pt3 -= pt1;
	pt4 -= pt2;

	float dl1 = abs(pt3.at<float>(0));
	float dl2 = abs(pt4.at<float>(0));

	pair<bool, bool> answer;

	answer.first = dl1 >= 1.5 && dl1 <= 1.6;
	answer.second = dl2 >= 1.5 && dl2 <= 1.6;

	return {answer.first, answer.second};
}

pair<vector<double *>, vector<Point2f>> get_rails_for_opt(pair<Vec4i, Vec4i>& rails, Camera& camera, Mat& plane,
	Mat& current, Mat& next)
{
	Mat cur = current.clone();
	Mat nex = next.clone();
	cvtColor(current, cur, COLOR_BGR2GRAY);
	cvtColor(next, nex, COLOR_BGR2GRAY);
	vector<uchar> status;
	vector<float> err;
	TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);

	vector<Point2f> pts_start, pts_tails;
	pts_start = {{static_cast<float>(rails.first[0]), static_cast<float>(rails.first[1])},
				 {static_cast<float>(rails.first[2]), static_cast<float>(rails.first[3])},
				 {static_cast<float>(rails.second[0]), static_cast<float>(rails.second[1])},
				 {static_cast<float>(rails.second[2]), static_cast<float>(rails.second[3])}};
	calcOpticalFlowPyrLK(cur, nex, pts_start, pts_tails, status, err, Size(15,15), 2, criteria);

	vector<double *> answer(4);
	vector<Point2f> good_pts(4);
	for(int i = 0; i < pts_start.size(); ++i)
		if(status[i] == 1)
		{
			auto* buffer = new double[3];
			Mat pt3d = general_estimate(pts_start[i], camera.R0, camera.t0,
				camera.K, plane);
			buffer[0] = pt3d.at<float>(0);
			buffer[1] = pt3d.at<float>(1);
			buffer[2] = pt3d.at<float>(2);
			good_pts[i] = pts_tails[i];
			answer[i] = buffer;
		}
		else
		{
			good_pts[i] = {-1000,-1000};
			answer[i] = NULL;
		}

	return {answer, good_pts};
}

