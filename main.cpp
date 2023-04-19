#include "reader.h"
#include "algos.h"
#include "final_tests.h"
#include <vector>
#include <algorithm>
using namespace std;

int main()
{
		string src = "/home/anreydron/Desktop/get.358/get.358_images/output_src/";
		string masks = "/home/anreydron/Desktop/get.358/get.358_images/output_segm/";
		const string time = "/home/anreydron/Desktop/get.358/get.358_images/Calib/frame_time.txt";
		const string speed = "/home/anreydron/Desktop/get.358/get.358_images/Calib/closest_speeds.txt";
		Reader reader(src, masks, time, speed, 1);
		std::vector<int> dynamic_classes = { 10, 11, 12, 13, 14, 15, 16, 17, 18 }; // значение в сегментационной маски динамических объектов
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
		const string prefix = "/home/anreydron/Desktop/get.358/test/"; // куда будем отписывать результаты (для обычно одометрии
		ofstream output_file1(prefix + "check_rails.txt");
		//no_optimized_odometry_on_descriptors(reader, camera, 1990, 5, dynamic_classes, output_file1, true, false);
		optimized_on_world_points_on_descriptors(reader, camera, 1990, 5, dynamic_classes, output_file1);
		SimpleStruct frame = reader.get_frame(5);
		cout << frame.current.rows << frame.current.cols;
		return 0;
}