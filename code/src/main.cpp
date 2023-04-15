#include "reader.h"
#include "algos.h"
#include "pointspreparation.h"
#include "final_tests.h"
#include <opencv2/calib3d.hpp>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

int main(void)
{
	// fs::directory_iterator& src, fs::directory_iterator& masks, const string& time, const string& speed
	fs::directory_iterator src_images("D:/TRAMWAY/get.358/output_src/"); // директория с исходными изображениями
	fs::directory_iterator masks("D:/TRAMWAY/get.358/output_segm/"); // директория с масками 
	fs::directory_iterator rail_masks("D:/TRAMWAY/content/rail_marking/output/"); // директория с масками рельс (не нужна пока - сами маски уже есть, но пока без них)
	//fs::directory_iterator rails_masks("D:/TRAMWAY/content/rail_marking/output/");
	const string time = "D:/TRAMWAY/get.358/Calib/frame_time.txt"; // txt файл с временем когда был получен кадр 
	const string speed = "D:/TRAMWAY/get.358/Calib/closest_speeds.txt"; // txt файл с модулем скоростей ближайших для каждого кадра по времени 
	Reader reader(src_images, masks, rail_masks, time, speed);
	std::vector<int> dynamic_classes = { 11,12,13,14,15,16,17,18 }; // значение в сегментационной маски динамических объектов 
	float Kdata[] = { 5.8101144196059124e+02, 0., 4.6611629315197757e+02, 0.,
	   5.8101144196059124e+02, 3.1452011177827347e+02, 0., 0., 1 }; // с calib для get.358
	Mat K(3, 3, cv::DataType<float>::type, Kdata);
	Vec3f R0data = { -9.6071019657095663e-02, 4.6384407919543125e-02,
	   -5.8740069123303252e-03 }; // с calib для get.358
	Mat t0(Size(1, 3), CV_32FC1);
	// с calib для get.358
	t0.at<float>(0, 0) = 3.3300000000000002e-01;
	t0.at<float>(0, 1) = 3.5999999999999999e-01; 
	t0.at<float>(0, 2) = 2.0869999995529653e+00;
	Mat R0;
	Rodrigues(R0data, R0);
	R0.convertTo(R0, CV_32FC1);
	// так как ск камеры не привычная XYZ, а z - Вперед, y - вправо, то осуществялем вращение вокруг оси X на 90 градусов 
	Mat rotation = Mat::eye(Size(3, 3), CV_32FC1);
	rotation.at<float>(1, 1) = 0;
	rotation.at<float>(2, 2) = 0;
	rotation.at<float>(1, 2) = 1;
	rotation.at<float>(2, 1) = -1;

	R0 = (R0 * rotation).inv(); // находим матрицу поворота от СК камеры к СК трамвая 
	t0 = -R0 * t0; // тоже самое для вектора 
	Camera camera = { K, R0, t0 };
	const string prefix = "D:/TRAMWAY/get.358/results_v_diplom/"; // куда будем отписывать результаты (для обычно одометрии 
	ofstream output_file1(prefix + "vanilla1.txt");
	//no_optimized_odometry_on_descriptors(reader, camera, 1990, 10, dynamic_classes, output_file1, true, false); 
	ofstream output_file3(prefix + "nightcall.txt"); // куда отписывать результаты оптимизации уже 
	optimized_on_world_points_on_descriptors(reader, camera, 1990, 5, dynamic_classes, output_file3);

	return 0;
}
