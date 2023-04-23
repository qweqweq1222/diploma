#pragma once
#include "algos.h"

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
			for (auto& match : kpm.matches)
			{
				float x_current = (float(kpm.kp1.at(match.queryIdx).pt.x));
				float y_current = (float(kpm.kp1.at(match.queryIdx).pt.y));
				float x_next = (float(kpm.kp2.at(match.trainIdx).pt.x));
				float y_next = (float(kpm.kp2.at(match.trainIdx).pt.y));

				if (masks)
				{

					if (find(dynamic_classes.begin(), dynamic_classes.end(), int(frame.mask.at<uchar>(int(y_current), int(x_current)))) != dynamic_classes.end())
						continue;
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

				if (find(dynamic_classes.begin(), dynamic_classes.end(), int(frame.mask.at<uchar>(int(v), int(u))))
					!= dynamic_classes.end())
					continue;

				start_points.emplace_back(Point2f(u, v));
				end_points.emplace_back(Point2f(x_next, y_next));
				if (int(frame.mask.at<uchar>(int(v), int(u))) == 0)
				{
					Mat pt_world_floor = general_estimate(Point(int(u), int(v)),
						camera.R0,
						camera.t0,
						camera.K,
						plane); // для точек с z = 0 оцениваем их положение в ск трамвая
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
			double angles_and_vecs_for_optimize[] =
				{ rt[0], rt[1], rt[2], rt[3], rt[4], rt[5] }; // начальное приближение для R,t

			//////////////////////////////			ОПТИМИЗАЦИЯ				////////////////////////////////

			/////////////////////////////           РЕЛЬСЫ                 ////////////////////////////////
			pair<Vec4i, Vec4i> rails = get_rails(frame.current, camera);
			vector<Point2f> pts_on_rails = {{static_cast<float>(rails.first[0]), static_cast<float>(rails.first[1])},
											{static_cast<float>(rails.first[2]),static_cast<float>(rails.first[3])},
											{static_cast<float>(rails.second[0]), static_cast<float>(rails.second[1])},
											{static_cast<float>(rails.second[2]), static_cast<float>(rails.second[3])}};
			vector<bool> is_valid = filter_rails(rails, camera, plane);
			pair<vector<double*>, vector<Point2f>>
				rails_for_opt = get_rails_for_opt(rails, camera, plane, frame.current, frame.next);

			/////////////////////////////           РЕЛЬСЫ                 ////////////////////////////////
			ceres::Problem problem;

			for (int m = 0; m < ground_points_3d_for_optimized.size(); ++m)
			{

				ceres::CostFunction* cost_function =
					SnavelyReprojectionErrorWorld::Create(double(image_points[m].x),
						double(image_points[m].y),
						camera.K.at<float>(0, 0),
						camera.K.at<float>(1, 1),
						camera.K.at<float>(0, 2),
						camera.K.at<float>(1, 2),
						camera.R0,
						camera.t0,
						frame.current.cols,
						frame.current.rows);
				problem.AddResidualBlock(cost_function,
					nullptr,
					angles_and_vecs_for_optimize,
					ground_points_3d_for_optimized[m]);

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
				for (int p = 0; p < 3; ++p)
					initial_point_pt[p] = ground_points_3d_for_optimized[m][p];
				for (int idx = 0; idx < 3;
					 ++idx) // ограничение на сколько можем от начального приближения уходить - +- 20% для точек на полу
				{
					if (idx != 2)
					{
						if (ground_points_3d_for_optimized[m][idx] >= 0)
						{
							problem.SetParameterLowerBound(ground_points_3d_for_optimized[m],
								idx,
								initial_point_pt[idx] * 0.80);
							problem.SetParameterUpperBound(ground_points_3d_for_optimized[m],
								idx,
								initial_point_pt[idx] * 1.20);
						}
						else
						{
							problem.SetParameterLowerBound(ground_points_3d_for_optimized[m],idx, initial_point_pt[idx] * 1.20);
							problem.SetParameterUpperBound(ground_points_3d_for_optimized[m], idx, initial_point_pt[idx] * 0.80);
						}
					}
					else
					{
						if (ground_points_3d_for_optimized[m][idx] >= 0)
						{
							problem.SetParameterLowerBound(ground_points_3d_for_optimized[m], idx, initial_point_pt[idx] - 0.2);
							problem.SetParameterUpperBound(ground_points_3d_for_optimized[m], idx, initial_point_pt[idx] + 0.2);
						}
						else
						{
							problem.SetParameterLowerBound(ground_points_3d_for_optimized[m], idx, initial_point_pt[idx] + 0.2);
							problem.SetParameterUpperBound(ground_points_3d_for_optimized[m], idx, initial_point_pt[idx] - 0.20);
						}
					}
				}
			}
			/////////////////////////////// рельсы
			// сначала добавляем reporjection error по рельсам
			for (int m = 0; m < rails_for_opt.first.size(); ++m)
			{
				if (rails_for_opt.first[m] != NULL)
				{
					ceres::CostFunction* cost_function =
						SnavelyReprojectionErrorWorld::Create(double(rails_for_opt.second[m].x),
							double(rails_for_opt.second[m].y),
							camera.K.at<float>(0, 0),
							camera.K.at<float>(1, 1),
							camera.K.at<float>(0, 2),
							camera.K.at<float>(1, 2),
							camera.R0,
							camera.t0,
							frame.current.cols,
							frame.current.rows);
					problem.AddResidualBlock(cost_function,
						nullptr,
						angles_and_vecs_for_optimize,
						rails_for_opt.first[m]);

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
					for (int p = 0; p < 3; ++p)
						initial_point_pt[p] = rails_for_opt.first[m][p];
					for (int idx = 0; idx < 3; ++idx)
					{
						if (idx != 2)
						{
							if (rails_for_opt.first[m][idx] >= 0)
							{
								problem.SetParameterLowerBound(rails_for_opt.first[m],
									idx,
									initial_point_pt[idx] * 0.80);
								problem.SetParameterUpperBound(rails_for_opt.first[m],
									idx,
									initial_point_pt[idx] * 1.20);
							}
							else
							{
								problem.SetParameterLowerBound(rails_for_opt.first[m],
									idx,
									initial_point_pt[idx] * 1.20);
								problem.SetParameterUpperBound(rails_for_opt.first[m],
									idx,
									initial_point_pt[idx] * 0.80);
							}
						}

						else
						{
							if (rails_for_opt.first[m][idx] >= 0)
							{
								problem.SetParameterLowerBound(rails_for_opt.first[m],
									idx,
									initial_point_pt[idx] - 0.2);
								problem.SetParameterUpperBound(rails_for_opt.first[m],
									idx,
									initial_point_pt[idx] + 0.2);
							}
							else
							{
								problem.SetParameterLowerBound(rails_for_opt.first[m],
									idx,
									initial_point_pt[idx] + 0.2);
								problem.SetParameterUpperBound(rails_for_opt.first[m],
									idx,
									initial_point_pt[idx] - 0.20);
							}
						}
					}
				}
			}


			// теперь добавляем регуляризатор, если он вообще есть

			for(int m = 0; m < is_valid.size(); ++m)
			{
				if (is_valid[m])
					if (rails_for_opt.first[m] != NULL && rails_for_opt.first[m + 2] != NULL)
					{
						ceres::CostFunction* cost_function =
							SnavelyReprojectionErrorWorld_::Create(double(rails_for_opt.second[m].x),
								double(rails_for_opt.second[m].y),
								camera.K.at<float>(0, 0),
								camera.K.at<float>(1, 1),
								camera.K.at<float>(0, 2),
								camera.K.at<float>(1, 2),
								camera.R0,
								camera.t0,
								frame.current.cols,
								frame.current.rows);
						problem.AddResidualBlock(cost_function,
							nullptr,
							rails_for_opt.first[m],
							rails_for_opt.first[m + 2]);

						double initial_point_pt[3];
						for (int p = 0; p < 3; ++p)
							initial_point_pt[p] = rails_for_opt.first[m][p];
						for (int idx = 0; idx < 3; ++idx)
						{
							if (idx != 2)
							{
								if (rails_for_opt.first[m][idx] >= 0)
								{
									problem.SetParameterLowerBound(rails_for_opt.first[m],
										idx,
										initial_point_pt[idx] * 0.80);
									problem.SetParameterUpperBound(rails_for_opt.first[m],
										idx,
										initial_point_pt[idx] * 1.20);
								}
								else
								{
									problem.SetParameterLowerBound(rails_for_opt.first[m],
										idx,
										initial_point_pt[idx] * 1.20);
									problem.SetParameterUpperBound(rails_for_opt.first[m],
										idx,
										initial_point_pt[idx] * 0.80);
								}
							}
							else
							{
								if (rails_for_opt.first[m][idx] >= 0)
								{
									problem.SetParameterLowerBound(rails_for_opt.first[m],
										idx,
										initial_point_pt[idx] - 0.2);
									problem.SetParameterUpperBound(rails_for_opt.first[m],
										idx,
										initial_point_pt[idx] + 0.2);
								}
								else
								{
									problem.SetParameterLowerBound(rails_for_opt.first[m],
										idx,
										initial_point_pt[idx] + 0.2);
									problem.SetParameterUpperBound(rails_for_opt.first[m],
										idx,
										initial_point_pt[idx] - 0.20);
								}
							}
						}

						for (int p = 0; p < 3; ++p)
							initial_point_pt[p] = rails_for_opt.first[m][p];
						for (int idx = 0; idx < 3; ++idx)
						{
							if (idx != 2)
							{
								if (rails_for_opt.first[m + 2][idx] >= 0)
								{
									problem.SetParameterLowerBound(rails_for_opt.first[m + 2],
										idx,
										initial_point_pt[idx] * 0.80);
									problem.SetParameterUpperBound(rails_for_opt.first[m + 2],
										idx,
										initial_point_pt[idx] * 1.20);
								}
								else
								{
									problem.SetParameterLowerBound(rails_for_opt.first[m + 2],
										idx,
										initial_point_pt[idx] * 1.20);
									problem.SetParameterUpperBound(rails_for_opt.first[m + 2],
										idx,
										initial_point_pt[idx] * 0.80);
								}
							}
							else
							{
								if (rails_for_opt.first[m + 2][idx] >= 0)
								{
									problem.SetParameterLowerBound(rails_for_opt.first[m + 2],
										idx,
										initial_point_pt[idx] - 0.2);
									problem.SetParameterUpperBound(rails_for_opt.first[m + 2],
										idx,
										initial_point_pt[idx] + 0.2);
								}
								else
								{
									problem.SetParameterLowerBound(rails_for_opt.first[m + 2],
										idx,
										initial_point_pt[idx] + 0.2);
									problem.SetParameterUpperBound(rails_for_opt.first[m + 2],
										idx,
										initial_point_pt[idx] - 0.20);
								}
							}
						}
					}

			}

			////////////////////////////// рельсы
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

			for(auto& pt : rails_for_opt.first)
				delete[] pt;

			cout << k << endl;
		}
	}
}
