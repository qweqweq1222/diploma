#pragma once
#include "reader.h"
#include "algos.h"
struct SnavelyReprojectionErrorWorld { // определяем нашу задачу квадратичной минимизации
	SnavelyReprojectionErrorWorld(double observed_x, double observed_y, double fx, double fy, double cx, double cy, cv::Mat R0, cv::Mat t0,
		double width, double height)
		: observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy), R0(R0), t0(t0), width(width), height(height) {}

	template <typename T>
	bool operator()(const T* const alpha_t, const T* const pt3d, T* residuals) const {


		T P3[3];
		T a = alpha_t[0];
		T b = alpha_t[1];
		T g = alpha_t[2];
		// CERES не особо позволяет внутрь себя что-то прокидывать, поэтому переход от ск трамвая (глобальной) в ск камеры и последующее нахождение проекции (точек на матрице камеры) пишем руками
		// здесь написано просто R0 * pt_3d + t0
		P3[0] = T(R0.at<float>(0,0))* pt3d[0] + T(R0.at<float>(0,1))* pt3d[1] + T(R0.at<float>(0,2))* pt3d[2] + T(t0.at<float>(0,0));
		P3[1] = T(R0.at<float>(1, 0)) * pt3d[0] + T(R0.at<float>(1, 1)) * pt3d[1] + T(R0.at<float>(1, 2)) * pt3d[2] + T(t0.at<float>(1, 0));
		P3[2] = T(R0.at<float>(2, 0)) * pt3d[0] + T(R0.at<float>(2, 1)) * pt3d[1] + T(R0.at<float>(2, 2)) * pt3d[2] + T(t0.at<float>(2, 0));
		// Находим координаты после смещения (перешли на предыдущем шаге в СК камеры) теперь осуществляем смещение к следующему кадру R * pt_camera + t
		P3[0] = T(cos(b) * cos(g)) * (P3[0]) - T(sin(g) * cos(b)) * (P3[1]) + T(sin(b)) * (P3[2]) + alpha_t[3];
		P3[1] = T(sin(a) * sin(b) * cos(g) + sin(g) * cos(a)) * (P3[0]) + T(cos(a) * cos(g) - sin(a) * sin(b) * sin(g)) * (P3[1]) - T(sin(a) * cos(b)) * (P3[2]) + alpha_t[4];
		P3[2] = T(sin(a) * sin(g) - sin(b) * cos(a) * cos(g)) * (P3[0]) + T(sin(a) * cos(g) + sin(b) * sin(g) * cos(a)) * (P3[1]) + T(cos(a) * cos(b)) * (P3[2]) + alpha_t[5];
		// находим проекцию на матрицу камеры
		T predicted_x = T(fx) * (P3[0]) / P3[2] + T(cx);
		T predicted_y = T(fy) * (P3[1]) / P3[2] + T(cy);
		// определяем вектор ошибок как:

		residuals[0] = (predicted_x - T(observed_x));
		residuals[1] = (predicted_y - T(observed_y));
		residuals[2] = predicted_x > T(0) && predicted_x < T(width) ? T(0) : T(1000);
		residuals[3] = predicted_y > T(0) && predicted_y < T(height) ? T(0) : T(1000);
		return true;
	}
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double fx, const double fy, const double cx, const double cy, const cv::Mat R0, const cv::Mat t0,
		const double width, const double height) {
		return (new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorWorld, 4, 6, 3>(
			new SnavelyReprojectionErrorWorld(observed_x, observed_y, fx, fy, cx, cy, R0, t0, width ,height)));
	}

	double observed_x;
	double observed_y;
	double width, height;
	double fx, fy, cx, cy;
	Mat R0, t0;
};


struct SnavelyReprojectionErrorWorld_ { // определяем нашу задачу квадратичной минимизации
	SnavelyReprojectionErrorWorld_(double observed_x, double observed_y, double fx, double fy, double cx, double cy, cv::Mat R0, cv::Mat t0,
		double width, double height)
		: observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy), R0(R0), t0(t0), width(width), height(height) {}

	template <typename T>
	bool operator()(const T* const pair1, const T* pair2, T* residuals) const {

		T d = (pair1[0] - pair2[0]) * (pair1[0] - pair2[0]) * (pair1[1] - pair2[1]) * (pair1[1] - pair2[1]) * (pair1[2] - pair2[2]) * (pair1[2] - pair2[2]);
		residuals[0] = T(10000) * (d - T(1.52 * 1.52));

		return true;
	}
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y, const double fx, const double fy, const double cx, const double cy, const cv::Mat R0, const cv::Mat t0,
		const double width, const double height) {
		return (new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorWorld_, 1, 3, 3>(
			new SnavelyReprojectionErrorWorld_(observed_x, observed_y, fx, fy, cx, cy, R0, t0, width ,height)));
	}

	double observed_x;
	double observed_y;
	double width, height;
	double fx, fy, cx, cy;
	Mat R0, t0;
};