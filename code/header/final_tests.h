#pragma once
#include "algos.h"

pair<vector<Point2f>, vector<Point2f>> get_flow(const vector<Point2f>& start, const Mat& current, const Mat& next) 
{
	Mat current_gray, next_gray;
	vector<Point2f> end;
	vector<Point2f> kp_current, kp_next;
	cvtColor(current, current_gray, COLOR_BGR2GRAY);
	cvtColor(next, next_gray, COLOR_BGR2GRAY);
	vector<uchar> status;
	vector<float> err;
	cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT)+(cv::TermCriteria::EPS), 10, 0.03);
	calcOpticalFlowPyrLK(current_gray, next_gray, start, end, status, err, Size(15, 15), 2, criteria);

	for (int i = 0; i < start.size(); ++i)
	{
		if (status[i] == 1)
		{
			kp_current.emplace_back(start[i]);
			kp_next.emplace_back(end[i]);
		}
	}
	return { kp_current, kp_next };
}

void no_optimized_odometry_on_descriptors(Reader& reader, Camera& camera, const int& number_of_iterations, const int& step,
	vector<int> dynamic_classes, ofstream& output_file, const bool masks = false, const bool local_dynamic = false) // masks - выкидывает по сегментационным маскам особые точки на динамических объектах
{
	// параметр local_dynamic - оставьте false
	Mat GLOBAL_COORDS = Mat::eye(Size(4, 4), CV_32FC1); 
	SimpleStruct frame;

	for (int k = 0; k < number_of_iterations; k += step)
	{
		Mat local = Mat::eye(Size(4, 4), CV_32FC1);
		frame = reader.get_frame(step);
		if (frame.speed > 0.1)
		{
			KeyPointMatches kpm = align_images(frame.current, frame.next);
			vector<Point2f> start, end;
			resize(frame.rails, frame.rails, Size(frame.current.cols, frame.current.rows), INTER_LINEAR);
			for (auto& match : kpm.matches)
			{
				float x_current = (float(kpm.kp1.at(match.queryIdx).pt.x));
				float y_current = (float(kpm.kp1.at(match.queryIdx).pt.y));
				float x_next = (float(kpm.kp2.at(match.trainIdx).pt.x));
				float y_next = (float(kpm.kp2.at(match.trainIdx).pt.y));

				if (masks)  // masks == true ?
				{

					if (find(dynamic_classes.begin(), dynamic_classes.end(), int(frame.mask.at<uchar>(int(y_current), int(x_current)))) != dynamic_classes.end())
						continue;
					if (local_dynamic) // local_dynamic == true ? 
					{
						if (!is_dynamic(frame.current, frame.mask, Point(x_current, y_current), dynamic_classes) && dynamic_classes.size() != 0
							&& int(frame.mask.at<uchar>(int(y_current), int(x_current))) != 0
							&& int(frame.mask.at<uchar>(int(y_current), int(x_current))) != 8)
							// int(frame.rails.at<uchar>(int(y_current), int(x_current))) == 255) // дополнительная проверка по маскам рельс - пока не используется 
						{
							start.push_back(Point2f(x_current, y_current));
							end.emplace_back(Point2f(x_next, y_next));

						}
						else
							continue;
					}
					else
					{
						start.push_back(Point2f(x_current, y_current));
						end.emplace_back(Point2f(x_next, y_next));
					}
				}

				else
				{
					start.emplace_back(Point2f(x_current, y_current));
					end.emplace_back(Point2f(x_next, y_next));
				}
			}

			Mat E, R, t, useless_masks;
			E = findEssentialMat(start, end, camera.K, RANSAC, 0.99, 1.0, useless_masks);
			cv::recoverPose(E, start, end, camera.K, R, t, useless_masks);

			t.convertTo(t, CV_32FC1);
			R.convertTo(R, CV_32FC1);
			// матрица R0 из main - это матрица перехода от СК камеры в СК трамвая. Чтобы перейти в глобальную систему координат нужно осуществить преобразование R^-1 * R * R 
			R = camera.R0.inv() * R * camera.R0; 
			t = camera.R0.inv() * t; // переводим вектор смещения в СК трамвая 
			t *= (0.001 * frame.dt * frame.speed); // учитываем модуль скорости 

			
			R.copyTo(local(Rect(0, 0, 3, 3)));
			t.copyTo(local(Rect(3, 0, 1, 3)));

			GLOBAL_COORDS *= local.inv(); // 
			output_file << GLOBAL_COORDS.at<float>(0, 3) << " " << GLOBAL_COORDS.at<float>(1, 3) << " " << GLOBAL_COORDS.at<float>(2, 3) << endl;

			cout << k << endl;
		}
	}
}

// НЕ ОБРАЩАЙТЕ ВНИМАНИЕ  
void no_optimized_odometry_on_optical_flow(Reader& reader, Camera& camera, const int& number_of_iterations, const int& step,
	vector<int> dynamic_classes, ofstream& output_file, const bool masks = false, const bool local_dynamic = false) 
{
	Mat GLOBAL_COORDS = Mat::eye(Size(4, 4), CV_32FC1); 
	SimpleStruct frame;

	for (int k = 0; k < number_of_iterations; k += step)
	{
		Mat local = Mat::eye(Size(4, 4), CV_32FC1);
		frame = reader.get_frame(step);
		Mat descriptors;
		vector<KeyPoint> keypoints_current;
		vector<Point2f> good_keypoints_current;
		vector<Point2f> start, end;
		Ptr<ORB> detector = ORB::create();
		resize(frame.rails, frame.rails, Size(frame.current.cols, frame.current.rows), INTER_LINEAR);
		detector->detectAndCompute(frame.current, noArray(), keypoints_current, descriptors); // íàõîäèì ôè÷è íà ïåðâîì èçîáðàæåíèè
		
		// òàêàÿ æå ëîãèêà, êàê è â ïðåäûäóùåì ìåòîäå
		if (masks)
		{
			for (int i = 0; i < keypoints_current.size(); ++i)
			{
				float x = keypoints_current[i].pt.x;
				float y = keypoints_current[i].pt.y;

				if (find(dynamic_classes.begin(), dynamic_classes.end(), int(frame.mask.at<uchar>(int(y), int(x)))) != dynamic_classes.end())
					continue;
				if (local_dynamic)
					if (!is_dynamic(frame.current, frame.mask, Point(x, y), dynamic_classes) && dynamic_classes.size() != 0 
						&& int(frame.mask.at<uchar>(int(y), int(x))) != 0 && int(frame.mask.at<uchar>(int(y), int(x))) != 8) // òî÷êà íå äèíàìè÷åñêàÿ - ïðîâåðÿåì îêðåñòíîñòü 
						good_keypoints_current.emplace_back(keypoints_current[i].pt);
					else
						continue;
				else
					good_keypoints_current.emplace_back(keypoints_current[i].pt);
			}
		}
		else
			for (auto& pt : keypoints_current)
				good_keypoints_current.emplace_back(pt.pt);

		pair<vector<Point2f>, vector<Point2f>> points = get_flow(good_keypoints_current, frame.current, frame.next);

		for (int j = 0; j < points.first.size(); ++j)
		{
			line(frame.current, points.first[j], points.second[j], Scalar(255, 0, 255), 3);
		}
		//imshow("frame", frame.current);
		//waitKey(0);
		Mat E, R, t, useless_masks;
		E = findEssentialMat(points.first, points.second, camera.K, RANSAC, 0.99, 1.0, useless_masks);
		recoverPose(E, points.first, points.second, camera.K, R, t, useless_masks);

		t.convertTo(t, CV_32FC1);
		R.convertTo(R, CV_32FC1);

		R = camera.R0.inv() * R * camera.R0; // ïåðåâîäèì ïîëó÷åííóþ ìàòðèöó ïîâîðîòà R â ñ.ê. ðîáîòà 
		t = camera.R0.inv() * t; // -/- ñ âåêòîðîì ñìåùåíèÿ
		t *= (0.001 * frame.dt * frame.speed); // äîìíîæàåì íà ìîäóëü ñêîðîñòè ñ IMU 

		// çàïèñûâàåì R,t â ìàòðöó [R|t] ðàçìåðà 4x4  
		R.copyTo(local(Rect(0, 0, 3, 3)));
		t.copyTo(local(Rect(3, 0, 1, 3)));

		GLOBAL_COORDS *= local.inv(); // îáðàòíàÿ òàê êàê recoverPose âîçâðàùàåò R è t íå èç current->next, à next->current
		output_file << GLOBAL_COORDS.at<float>(0, 3) << " " << GLOBAL_COORDS.at<float>(1, 3) << " " << GLOBAL_COORDS.at<float>(2, 3) << endl;

		cout << k << endl;
	}
}



// НЕ ОБРАЩАЙТЕ ВНИМАНИЕ
void optimized_on_camera_points_descriptors(Reader& reader, Camera& camera, const int& number_of_iterations, const int& step,
	vector<int> dynamic_classes, ofstream& output_file)
{
	Mat GLOBAL_COORDS = Mat::eye(Size(4, 4), CV_32FC1); // ñþäà áóäåì "ñêëàäûâàòü" ïîñëåäîâàòåëüíî ðåçóëüòàòû ñìåùåíèé 
	SimpleStruct frame;
	Mat plane(Size(4, 1), CV_32FC1); // z = 0;
	plane.at<float>(0, 0) = 0;
	plane.at<float>(0, 1) = 0;
	plane.at<float>(0, 2) = 1;
	plane.at<float>(0, 3) = 0;

	for (int k = 0; k < number_of_iterations; k += step)
	{
		Mat local = Mat::eye(Size(4, 4), CV_32FC1);
		frame = reader.get_frame(step);
		if (frame.speed > 0.1)
		{
			KeyPointMatches kpm = align_images(frame.current, frame.next);
			vector<Point2f> start, end;
			vector<Point2f> ground_points_next; // ÂÍÈÌÀÍÈÅ: ýòî òî÷êè íà èçîáðàæåíèè 
			vector<Vec3f> ground_points_current; // ÂÍÈÌÀÍÈÅ: ýòî òî÷êè â R3 â Ñ.Ê. êàìåðû
			for (auto& match : kpm.matches)
			{
				float x_current = (float(kpm.kp1.at(match.queryIdx).pt.x));
				float y_current = (float(kpm.kp1.at(match.queryIdx).pt.y));
				float x_next = (float(kpm.kp2.at(match.trainIdx).pt.x));
				float y_next = (float(kpm.kp2.at(match.trainIdx).pt.y));

				if (find(dynamic_classes.begin(), dynamic_classes.end(), int(frame.mask.at<uchar>(int(y_current), int(x_current)))) != dynamic_classes.end())
					continue;
				if (!is_dynamic(frame.current, frame.mask, Point(x_current, y_current), dynamic_classes))
				{
					start.push_back(Point2f(x_current, y_current));
					end.emplace_back(Point2f(x_next, y_next));
					if (int(frame.mask.at<uchar>(int(y_current), int(x_current))) == 0) // åñëè òî÷êà ëåæèò íà çåìëå è â îêðåñòíîñòè íåòó äèíàìè÷åñêèõ òî÷åê
					{
						// ïî ôàêòó, çäåñü ñîáèðàåì òî÷êè äëÿ îïòèìèçàöèè, ïîòîìó ÷òî ìîæåì îöåíèòü èõ ïîëîæåíèå â R3 èç-çà z=0
						ground_points_next.emplace_back(Point2f(x_current, y_current));
						Mat point3d = estimate(Point(x_current, y_current), camera.R0, camera.t0, camera.K);
						ground_points_current.emplace_back(Vec3f(point3d.at<float>(0, 0), point3d.at<float>(1, 0), point3d.at<float>(2, 0)));
					}
				}
				else
					continue;
			}
			Mat E, R, t, useless_masks;
			E = findEssentialMat(start, end, camera.K, RANSAC, 0.99, 1.0, useless_masks);
			recoverPose(E, start, end, camera.K, R, t, useless_masks);
			t.convertTo(t, CV_32FC1);
			R.convertTo(R, CV_32FC1);
			t *= (0.001 * frame.dt * frame.speed);
			R.copyTo(local(Rect(0, 0, 3, 3)));
			t.copyTo(local(Rect(3, 0, 1, 3)));

			vector<double> rt = get_angles_and_vec(local); // âûäåëèëè âåêòîð {óãîë; óãîë; óãîë; tx; ty; tz}
			double angles_and_vecs_for_optimize[] = { rt[0], rt[1], rt[2], rt[3], rt[4], rt[5] };

			vector<double*> ground_points_3d_for_optimized; // òî÷êè äëÿ îïòèìèçàöèè (ÂÍÈÌÀÍÈÅ: â Ñ.Ê. êàìåðû, à íå òðàìâàÿ)
			for (auto& vec : ground_points_current)
			{
				double* buffer = new double[3];
				buffer[0] = vec[0];
				buffer[1] = vec[1];
				buffer[2] = vec[2];
				ground_points_3d_for_optimized.emplace_back(buffer);
			}


			ceres::Problem problem;
			for (int k = 0; k < ground_points_3d_for_optimized.size(); ++k)
			{
				ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(double(ground_points_next[k].x), double(ground_points_next[k].y),
					camera.K.at<float>(0, 0), camera.K.at<float>(1, 1), camera.K.at<float>(0, 2), camera.K.at<float>(1, 2), frame.current.cols, frame.current.rows);
				problem.AddResidualBlock(cost_function, nullptr, angles_and_vecs_for_optimize, ground_points_3d_for_optimized[k]);

				// ôèêñèðóåì èçíà÷àëüíûå çíà÷åíèÿ 
				double initial_point_movement[6];
				double initial_point_pt[3];
				for (int m = 0; m < 3; ++m)
					initial_point_pt[m] = ground_points_3d_for_optimized[k][m];
				for (int m = 0; m < 6; ++m)
					initial_point_movement[m] = angles_and_vecs_for_optimize[m];
				// îãðàíè÷åíèÿ íà óãëû è âåêòîð ñìåùåíèÿ 
				for (int idx = 0; idx < 6; ++idx)
				{
					if (initial_point_movement[idx] >= 0)
					{
						problem.SetParameterLowerBound(angles_and_vecs_for_optimize, idx, initial_point_movement[idx] * 0.80);
						problem.SetParameterUpperBound(angles_and_vecs_for_optimize, idx, initial_point_movement[idx] * 1.20);
					}
					else
					{
						problem.SetParameterLowerBound(angles_and_vecs_for_optimize, idx, initial_point_movement[idx] * 1.20);
						problem.SetParameterUpperBound(angles_and_vecs_for_optimize, idx, initial_point_movement[idx] * 0.80);
					}
				}
				// îãðàíè÷åíèÿ íà êîîðäèíàòû
				for (int idx = 0; idx < 3; ++idx)
				{
					if (initial_point_pt[idx] >= 0)
					{
						problem.SetParameterLowerBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 0.80);
						problem.SetParameterUpperBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 1.20);
					}
					else
					{
						problem.SetParameterLowerBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 1.20);
						problem.SetParameterUpperBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 0.80);
					}
				}
			}
			// ðåøàåì ïðîáëåìó
			ceres::Solver::Options options;
			options.linear_solver_type = ceres::DENSE_SCHUR;
			options.minimizer_progress_to_stdout = true;
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);

			///////////////////////////			ÎÏÒÈÌÈÇÀÖÈß				///////////////////////////////////

			////////////////////////// ÏÅÐÅÕÎÄ Â ÃËÎÁÀËÜÍÓÞ ÑÊ          //////////////////////////////////

			local = reconstruct_from_v6(angles_and_vecs_for_optimize);

			local(Rect(0, 0, 3, 3)).copyTo(R);
			local(Rect(3, 0, 1, 3)).copyTo(t);

			R = camera.R0.inv() * R * camera.R0;
			t = camera.R0.inv() * t;

			local = Mat::eye(Size(4, 4), CV_32FC1);

			R.copyTo(local(Rect(0, 0, 3, 3)));
			t.copyTo(local(Rect(3, 0, 1, 3)));

			////////////////////////// ÏÅÐÅÕÎÄ Â ÃËÎÁÀËÜÍÓÞ ÑÊ          //////////////////////////////////

			GLOBAL_COORDS *= local.inv();
			output_file << GLOBAL_COORDS.at<float>(0, 3) << " " << GLOBAL_COORDS.at<float>(1, 3) << " " << GLOBAL_COORDS.at<float>(2, 3) << endl;

			for (auto& pt : ground_points_3d_for_optimized)
				delete[] pt;
			cout << k << endl;
		}
	}
}


// НЕ ОБРАЩАЙТЕ ВНИМАНИЕ 
void optimized_on_camera_points_on_flow(Reader& reader, Camera& camera, const int& number_of_iterations, const int& step,
	vector<int> dynamic_classes, ofstream& output_file)
{
	Mat GLOBAL_COORDS = Mat::eye(Size(4, 4), CV_32FC1); 
	SimpleStruct frame;
	Mat plane(Size(4, 1), CV_32FC1); // z = 0;
	plane.at<float>(0, 0) = 0;
	plane.at<float>(0, 1) = 0;
	plane.at<float>(0, 2) = 1;
	plane.at<float>(0, 3) = 0;

	for (int k = 0; k < number_of_iterations; k += step)
	{
		Mat local = Mat::eye(Size(4, 4), CV_32FC1);
		frame = reader.get_frame(step);
		Mat descriptors;
		vector<KeyPoint> keypoints_current;
		vector<Point2f> good_keypoints_current;
		vector<Point2f> start, end;
		Ptr<KAZE> detector = KAZE::create();
		vector<Vec3f> ground_points_current;
		vector<Point2f> ground_points_next;
		detector->detectAndCompute(frame.current, noArray(), keypoints_current, descriptors);
		for (int i = 0; i < keypoints_current.size(); ++i)
		{
			float x_current = keypoints_current[i].pt.x;
			float y_current = keypoints_current[i].pt.y;

			if (find(dynamic_classes.begin(), dynamic_classes.end(), int(frame.mask.at<uchar>(int(y_current), int(x_current)))) != dynamic_classes.end())
				continue;
			if (!is_dynamic(frame.current, frame.mask, Point(x_current, y_current), dynamic_classes) && int(frame.mask.at<uchar>(int(y_current), int(x_current))) != 8)
				good_keypoints_current.emplace_back(keypoints_current[i].pt);
		}

		pair<vector<Point2f>, vector<Point2f>> points = get_flow(good_keypoints_current, frame.current, frame.next);
		for (int i = 0; i < points.second.size(); ++i)
		{
			float x_current = points.first[i].x;
			float y_current = points.first[i].y;
			float x_next = points.second[i].x;
			float y_next = points.second[i].y;

			float dx = x_next - x_current;
			float dy = y_next - y_current;
			if (sqrt(pow(dx, 2) + pow(dy, 2)) < 20 && sqrt(pow(dx, 2) + pow(dy, 2)) > 15)
			{
				if (int(frame.mask.at<uchar>(int(y_current), int(x_current))) == 0)
				{
					Mat pt_world = general_estimate(Point(int(x_current), int(y_current)), camera.R0, camera.t0, camera.K, plane); // pt3d óæå äëÿ ïåðåâîäà â R3 â world
					if (pt_world.at<float>(0, 0) > -5 && pt_world.at<float>(0, 0) < 5)
					{
						Mat point3d = estimate(Point(points.first[i].x, points.first[i].y), camera.R0, camera.t0, camera.K);
						ground_points_current.emplace_back(Vec3f(point3d.at<float>(0, 0), point3d.at<float>(1, 0), point3d.at<float>(2, 0)));
						ground_points_next.emplace_back(Point2f(points.second[i].x, points.second[i].y));
						line(frame.current, points.first[i], points.second[i], Scalar(255, 0, 255), 3);
					}
				}
			}
		}
		
		imshow("frame", frame.current);
		waitKey(0);
		Mat E, R, t, useless_masks;
		E = findEssentialMat(points.first, points.second, camera.K, RANSAC, 0.99, 1.0, useless_masks);
		recoverPose(E, points.first, points.second, camera.K, R, t, useless_masks);
		t.convertTo(t, CV_32FC1);
		R.convertTo(R, CV_32FC1);
		t *= (0.001 * frame.dt * frame.speed);
		R.copyTo(local(Rect(0, 0, 3, 3)));
		t.copyTo(local(Rect(3, 0, 1, 3)));

		vector<double> rt = get_angles_and_vec(local); // âûäåëèëè âåêòîð {óãîë; óãîë; óãîë; tx; ty; tz}
		double angles_and_vecs_for_optimize[] = { rt[0], rt[1], rt[2], rt[3], rt[4], rt[5] };

		vector<double*> ground_points_3d_for_optimized; // òî÷êè äëÿ îïòèìèçàöèè (ÂÍÈÌÀÍÈÅ: â Ñ.Ê. êàìåðû, à íå òðàìâàÿ)
		for (auto& vec : ground_points_current)
		{
			double* buffer = new double[3];
			buffer[0] = vec[0];
			buffer[1] = vec[1];
			buffer[2] = vec[2];
			ground_points_3d_for_optimized.emplace_back(buffer);
		}


		ceres::Problem problem;
		for (int k = 0; k < ground_points_3d_for_optimized.size(); ++k)
		{
			ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(double(ground_points_next[k].x), double(ground_points_next[k].y),
				camera.K.at<float>(0, 0), camera.K.at<float>(1, 1), camera.K.at<float>(0, 2), camera.K.at<float>(1, 2), frame.current.cols, frame.current.rows);
			problem.AddResidualBlock(cost_function, nullptr, angles_and_vecs_for_optimize, ground_points_3d_for_optimized[k]);

			// ôèêñèðóåì èçíà÷àëüíûå çíà÷åíèÿ 
			double initial_point_movement[6];
			double initial_point_pt[3];
			for (int m = 0; m < 3; ++m)
				initial_point_pt[m] = ground_points_3d_for_optimized[k][m];
			for (int m = 0; m < 6; ++m)
				initial_point_movement[m] = angles_and_vecs_for_optimize[m];
			// îãðàíè÷åíèÿ íà óãëû è âåêòîð ñìåùåíèÿ 
			for (int idx = 0; idx < 6; ++idx)
			{
				if (initial_point_movement[idx] >= 0)
				{
					problem.SetParameterLowerBound(angles_and_vecs_for_optimize, idx, initial_point_movement[idx] * 0.90);
					problem.SetParameterUpperBound(angles_and_vecs_for_optimize, idx, initial_point_movement[idx] * 1.10);
				}
				else
				{
					problem.SetParameterLowerBound(angles_and_vecs_for_optimize, idx, initial_point_movement[idx] * 1.10);
					problem.SetParameterUpperBound(angles_and_vecs_for_optimize, idx, initial_point_movement[idx] * 0.90);
				}
			}
			// îãðàíè÷åíèÿ íà êîîðäèíàòû
			for (int idx = 0; idx < 3; ++idx)
			{
				if (initial_point_pt[idx] >= 0)
				{
					problem.SetParameterLowerBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 0.10);
					problem.SetParameterUpperBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 1.90);
				}
				else
				{
					problem.SetParameterLowerBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 1.10);
					problem.SetParameterUpperBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 0.90);
				}
			}
		}
		// ðåøàåì ïðîáëåìó
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_SCHUR;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		///////////////////////////			ÎÏÒÈÌÈÇÀÖÈß				///////////////////////////////////

		////////////////////////// ÏÅÐÅÕÎÄ Â ÃËÎÁÀËÜÍÓÞ ÑÊ          //////////////////////////////////

		local = reconstruct_from_v6(angles_and_vecs_for_optimize);

		local(Rect(0, 0, 3, 3)).copyTo(R);
		local(Rect(3, 0, 1, 3)).copyTo(t);

		R = camera.R0.inv() * R * camera.R0;
		t = camera.R0.inv() * t;

		local = Mat::eye(Size(4, 4), CV_32FC1);

		R.copyTo(local(Rect(0, 0, 3, 3)));
		t.copyTo(local(Rect(3, 0, 1, 3)));

		////////////////////////// ÏÅÐÅÕÎÄ Â ÃËÎÁÀËÜÍÓÞ ÑÊ          //////////////////////////////////

		GLOBAL_COORDS *= local.inv();
		output_file << GLOBAL_COORDS.at<float>(0, 3) << " " << GLOBAL_COORDS.at<float>(1, 3) << " " << GLOBAL_COORDS.at<float>(2, 3) << endl;

		for (auto& pt : ground_points_3d_for_optimized)
			delete[] pt;
		cout << k << endl;
	}
}
// ОДОМЕТРИЯ С ОПТИМИЗАЦИЕЙ И РЕГУЛЯРИЗАТОРОМ (ОГРАНИЧЕНИЯМИ НА ТОЧКИ В НУЛЕВОЙ ПЛОСКОСТИ (ПОКА ОТДЕЛЬНО НЕ ТРЕКАЕМ ТОЧКИ НА РЕЛЬСАХ))
void optimized_on_world_points_on_descriptors(Reader& reader, Camera& camera, const int& number_of_iterations, const int& step,
	vector<int> dynamic_classes, ofstream& output_file)
{
	Mat GLOBAL_COORDS = Mat::eye(Size(4, 4), CV_32FC1);
	SimpleStruct frame;

	Mat plane(Size(4, 1), CV_32FC1); // определяем матрицу A B C D плоскости z = 0 для метода general_estimate;
	plane.at<float>(0, 0) = 0;
	plane.at<float>(0, 1) = 0;
	plane.at<float>(0, 2) = 1;
	plane.at<float>(0, 3) = 0;

	for (int k = 0; k < number_of_iterations; k += step)
	{
		vector<Mat> floor_points; // точки на поверхности земли (определяем по сегментационным маскам пока - без рельс )
		Mat local = Mat::eye(Size(4, 4), CV_32FC1);
		frame = reader.get_frame(step); // читаем фрейм с заданнаым шагом (смотри reader)
		if (frame.speed > 0.1)
		{
			KeyPointMatches kpm = align_images(frame.current, frame.next, 1000);
			vector<Point2f> image_points;
			vector<Point2f> start_points, end_points;
			for (auto& match : kpm.matches)
			{
				float u = (float(kpm.kp1.at(match.queryIdx).pt.x));
				float v = (float(kpm.kp1.at(match.queryIdx).pt.y));
				float x_next = (float(kpm.kp2.at(match.trainIdx).pt.x));
				float y_next = (float(kpm.kp2.at(match.trainIdx).pt.y));

				if (find(dynamic_classes.begin(), dynamic_classes.end(), int(frame.mask.at<uchar>(int(v), int(u)))) != dynamic_classes.end())
					continue;
				
				start_points.emplace_back(Point2f(u, v));
				end_points.emplace_back(Point2f(x_next, y_next));
				if (int(frame.mask.at<uchar>(int(v), int(u))) == 0) 
				{
					Mat pt_world_floor = general_estimate(Point(int(u), int(v)), camera.R0, camera.t0, camera.K, plane); // для точек с z = 0 оцениваем их положение в ск трамвая 
					floor_points.emplace_back(pt_world_floor); 
					image_points.emplace_back(Point2f(x_next, y_next)); 
				}
			}
			vector<double*> ground_points_3d_for_optimized; // CERES принимает на вход double* - поэтому копируем все данные с floor_points 
			for (auto& pt : floor_points)
			{
				double* buffer = new double[3]; 
				buffer[0] = pt.at<float>(0, 0);
				buffer[1] = pt.at<float>(1, 0);
				buffer[2] = pt.at<float>(2, 0);
				ground_points_3d_for_optimized.emplace_back(buffer);
			}
			Mat E, R, t, useless_masks;
			E = findEssentialMat(start_points, end_points, camera.K, RANSAC, 0.99, 1.0, useless_masks);
			recoverPose(E, start_points, end_points, camera.K, R, t, useless_masks); // находим R, t в СК КАМЕРЫ ! 
			t.convertTo(t, CV_32FC1);
			R.convertTo(R, CV_32FC1);
			t *= (0.001 * frame.dt * frame.speed);
			R.copyTo(local(Rect(0, 0, 3, 3)));
			t.copyTo(local(Rect(3, 0, 1, 3)));

			vector<double> rt = get_angles_and_vec(local); 
			double angles_and_vecs_for_optimize[] = { rt[0], rt[1], rt[2], rt[3], rt[4] , rt[5] }; // начальное приближение для R,t 
			
			//////////////////////////////			ОПТИМИЗАЦИЯ				////////////////////////////////
			ceres::Problem problem;

			for (int k = 0; k < ground_points_3d_for_optimized.size(); ++k)
			{
				ceres::CostFunction* cost_function = SnavelyReprojectionErrorWorld::Create(double(image_points[k].x), double(image_points[k].y),
					camera.K.at<float>(0, 0), camera.K.at<float>(1, 1), camera.K.at<float>(0, 2), camera.K.at<float>(1, 2), camera.R0, camera.t0);
				problem.AddResidualBlock(cost_function, nullptr, angles_and_vecs_for_optimize, ground_points_3d_for_optimized[k]);


				double start[6];
				for (int p = 0; p < 6; ++p)
					start[p] = angles_and_vecs_for_optimize[p];

				for (int p = 0; p < 6; ++p) // ограничение на сколько можем от начального приближения уходить - +- 20% для углов и смещений 
				{
					if (start[p] >= 0)
					{
						problem.SetParameterLowerBound(angles_and_vecs_for_optimize, p, start[p] * 0.80);
						problem.SetParameterUpperBound(angles_and_vecs_for_optimize, p, start[p] * 1.20);
					}
					else
					{
						problem.SetParameterLowerBound(angles_and_vecs_for_optimize, p, start[p] * 1.20);
						problem.SetParameterUpperBound(angles_and_vecs_for_optimize, p, start[p] * 0.80);
					}
				}
				double initial_point_pt[3];
				for (int m = 0; m < 3; ++m)
					initial_point_pt[m] = ground_points_3d_for_optimized[k][m];
				for (int idx = 0; idx < 3; ++idx) // ограничение на сколько можем от начального приближения уходить - +- 20% для точек на полу 
				{
					if (ground_points_3d_for_optimized[k][idx] >= 0)
					{
						problem.SetParameterLowerBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 0.80);
						problem.SetParameterUpperBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 1.20);
					}
					else
					{
						problem.SetParameterLowerBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 1.20);
						problem.SetParameterUpperBound(ground_points_3d_for_optimized[k], idx, initial_point_pt[idx] * 0.80);
					}
				}

				// дополнительное ограничение на то, что на земле не выше 15 см можем лежать (далее можно тестировать с точками на рельсах) 
				problem.SetParameterLowerBound(ground_points_3d_for_optimized[k], 2, 0);
				problem.SetParameterUpperBound(ground_points_3d_for_optimized[k], 2, 0.15);
			}
			
			
			ceres::Solver::Options options;
			options.linear_solver_type = ceres::DENSE_SCHUR; // метод решения 
			options.minimizer_progress_to_stdout = true; // показываем результаты в консоль 
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);

		
			local = reconstruct_from_v6(angles_and_vecs_for_optimize); // восстанавливаем матрицу [R|t]

			local(Rect(0, 0, 3, 3)).copyTo(R);
			local(Rect(3, 0, 1, 3)).copyTo(t);

			R = camera.R0.inv() * R * camera.R0; // Переводим матрицу из СК камеры в СК трамвая 
			t = camera.R0.inv() * t; // тоже самое с вектором смещения 

			local = Mat::eye(Size(4, 4), CV_32FC1);

			R.copyTo(local(Rect(0, 0, 3, 3)));
			t.copyTo(local(Rect(3, 0, 1, 3)));

			

			GLOBAL_COORDS *= local.inv();
			// отписываем результат
			output_file << GLOBAL_COORDS.at<float>(0, 3) << " " << GLOBAL_COORDS.at<float>(1, 3) << " " << GLOBAL_COORDS.at<float>(2, 3) << endl;
			
			//чистим память 
			for (auto& pt : ground_points_3d_for_optimized)
				delete[] pt;
			cout << k << endl;
		}
	}
}
