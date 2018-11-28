#include "PerformanceCheck.h"

namespace GPerformance {
	bool Func_exist = true;

	void InitMatrix(float* A, int r_s, int c_s)
	{
		for (int r = 0; r < r_s; r++)
			for (int c = 0; c < c_s; c++)
				A[r * c_s + c] = 0.000123f;
	}

	//----------------------------------------------------------------
	void GPU_mul(float* A, float* B, float* C, int r_s, int c_s)
	{
		int size = c_s;
		concurrency::array_view<const float, 2> a(r_s, c_s, A);
		concurrency::array_view<const float, 2> b(r_s, c_s, B);
		concurrency::array_view<float, 2> c(r_s, c_s, C);

		concurrency::parallel_for_each(c.extent, [=](concurrency::index<2> idx) restrict(amp)
		{
			int row = idx[0];
			int col = idx[1];
			float sum = 0.0f;

			for (int i = 0; i < size; i++)
			{
				sum += a(row, i) * b(i, col);
			}

			c[idx] = sum;
		}
		);
		c.synchronize();
		Func_exist = true;
	}

	void CPU_PARALLEL_mul(float* A, float* B, float* C, int r_s, int c_s)
	{
		size_t size_n = size_t(r_s), size_m = size_t(c_s), size_w = size_t(c_s);
		concurrency::parallel_for(size_t(0), size_n, [&](size_t i)
		{
			for (size_t j = 0; j < size_w; j++)
			{
				float sum = 0.0f;
				for (size_t k = 0; k < size_m; k++)
				{
					sum += A[i * size_m + k] * B[k * size_w + i];
				}
				C[i * size_w + j] = sum;
			}
		});
		Func_exist = true;
	}

	void CPU_mul(float* A, float* B, float* C, int r_s, int c_s)
	{
		for (int i = 0; i < r_s; i++)
		{
			for (int j = 0; j < c_s; j++)
			{
				float sum = 0.0f;
				for (int k = 0; k < c_s; k++)
				{
					sum += A[i * c_s + k] * B[k * c_s + j];
				}
				C[i * c_s + j] = sum;
			}
		}
		Func_exist = true;
	}
	//----------------------------------------------------------------
	void GPU_plus_Val(float* A, float* B, float* C, int r_s, int c_s)
	{
		concurrency::array_view<const float, 1> other(r_s * c_s, A);
		concurrency::array_view<float, 1> target(r_s * c_s, B);

		parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
		{
			target[idx] += other[idx];
		});
		target.synchronize();
		Func_exist = true;
	}

	void CPU_PARALLEL_plus_Val(float* A, float* B, float* C, int r_s, int c_s)
	{
		std::cout << "There isn't implementation for CPU Parallel." << std::endl;
		Func_exist = false;
	}

	void CPU_plus_Val(float* A, float* B, float* C, int r_s, int c_s)
	{
		float Val = 0.023f;
		for (int r = 0; r < r_s; r++)
		{
			for (int c = 0; c < c_s; c++)
			{
				A[r * r_s + c] += Val;
			}
		}
		Func_exist = true;
	}
	//----------------------------------------------------------------
	void GPU_ReLU(float* A, float* B, float* C, int r_s, int c_s)
	{
		concurrency::array_view<const float, 1> other(r_s * c_s, A);
		concurrency::array_view<float, 1> target(r_s * c_s, B);
		concurrency::parallel_for_each(target.extent, [=](concurrency::index<1> idx) restrict(amp)
		{
			target[idx] = (other[idx] > 0) ? other[idx] : 0.0f;
		});
		target.synchronize();
		Func_exist = true;
	}

	void CPU_PARALLEL_ReLU(float* A, float* B, float* C, int r_s, int c_s)
	{
		size_t size_n = size_t(r_s), size_w = size_t(c_s);
		concurrency::parallel_for(size_t(0), size_n, [&](size_t i)
		{
			for (size_t j = 0; j < size_w; j++)
			{
				A[i * size_w + j] = (B[i * size_w + j] > 0) ? B[i * size_w + j] : 0.0f;
			}
		});
		Func_exist = true;
	}

	void CPU_ReLU(float* A, float* B, float* C, int r_s, int c_s)
	{
		std::cout << "There isn't implementation for CPU." << std::endl;
		Func_exist = false;
	}
	//----------------------------------------------------------------
	void GPU_conv(float* A, float* B, float* C, int r_s, int c_s)
	{
		int radius = Filter_size_R / 2;
		int row = r_s, col = c_s;
		concurrency::extent<2> e(r_s, c_s);
		concurrency::array_view<const float, 2> a(Filter_size_R, Filter_size_R, A);
		concurrency::array_view<const float, 2> b(e, B);
		concurrency::array_view<float, 2> c(e, C);

		concurrency::parallel_for_each(b.extent, [=](concurrency::index<2> idx) restrict(amp)
		{
			float sum = 0.0f;
			for (int r = -radius; r <= radius; r++)
			{
				for (int c = -radius; c <= radius; c++)
				{
					if ((idx[0] + r >= 0) && (idx[1] + c >= 0) && (idx[0] + r < row) && (idx[1] + c < col))
					{
						sum += a[r + radius][c + radius] * b[idx[0] + r][idx[1] + c];
					}
				}
			}
			c[idx] = sum;
		});
		c.synchronize();
		Func_exist = true;
	}

	void CPU_PARALLEL_conv(float* A, float* B, float* C, int r_s, int c_s)
	{
		int radius = Filter_size_R / 2;
		int row = r_s, col = c_s;
		float* result = C;
		float* filter = A;
		int filter_size = Filter_size_R;
		size_t size = 1;
		int lowlimit = radius;
		int upperlimit = radius;
		concurrency::parallel_for(size_t(0), size, [&](size_t i)
		{
			float* tmp_img = B;
			for (int y = lowlimit; y < (row - upperlimit); y++)//single input
			{
				for (int x = lowlimit; x < (col - upperlimit); x++)//single input
				{
					float sum = 0.0f;
					for (int p = -radius; p <= radius; p++)//filter kernal (rows)
					{
						for (int q = -radius; q <= radius; q++)//filter kernal (cols)
						{
							if ((y + p >= 0) && (x + q >= 0) && (y + p < row) && (x + q < col))
							{
								sum += filter[(p + radius) * filter_size + (q + radius)] * tmp_img[(y + p) * col + x + q];
							}
						}
					}
					result[(y - lowlimit) * (col - 2 * lowlimit) + (x - lowlimit)] += sum;
				}
			}
		});
		Func_exist = true;
	}

	void CPU_conv(float* A, float* B, float* C, int r_s, int c_s)
	{
		std::cout << "There isn't implementation for CPU." << std::endl;
		Func_exist = false;
	}
	//----------------------------------------------------------------
	void SingleRun(float* A, float* B, float* C, int r_s, int c_s, int index)
	{
		switch (index)
		{
		case 0: {PrintTime(*CPU_mul, *CPU_PARALLEL_mul, *GPU_mul, A, B, C, r_s, c_s);					break; }
		case 1: {PrintTime(*CPU_plus_Val, *CPU_PARALLEL_plus_Val, *GPU_plus_Val, A, B, C, r_s, c_s);	break; }
		case 2: {PrintTime(*CPU_ReLU, *CPU_PARALLEL_ReLU, *GPU_ReLU, A, B, C, r_s, c_s);				break; }
		case 3: {PrintTime(*CPU_conv, *CPU_PARALLEL_conv, *GPU_conv, A, B, C, r_s, c_s);				break; }
		default: {break; }
		}
	}

	void PrintTime(void(*CPU_Func)(float*, float*, float*, int, int), void(*CPU_PARALLEL_Func)(float*, float*, float*, int, int),
		void(*GPU_Func)(float*, float*, float*, int, int), float* A, float* B, float* C, int r_s, int c_s)
	{
		int num = r_s, power_num = -5;
		while (num > 0)
		{
			num = num/2;
			power_num++;
		}

		std::vector<__int64> CPU_Vec(power_num, _I64_MAX/2);
		std::vector<__int64> CPU_PARALLEL_Vec(power_num, _I64_MAX/2);
		std::vector<__int64> GPU_Vec(power_num, _I64_MAX/2);
		__int64 begin;
		int counter = 0;
		for (int r = 32, c = 32; r <= r_s && c <= c_s; r*=2, c*=2)
		{
			std::cout << "==========================================" << std::endl << std::endl;
			std::cout << "row size: " << r << " , col size: " << c << std::endl;

			begin = GetTickCount();
			(*CPU_Func)(A, B, C, r, c);
			if (Func_exist == true)
			{
				CPU_Vec.at(counter) = GetTickCount() - begin;
				std::cout << "CPU time: " << CPU_Vec.at(counter) << std::endl;
			}
			else
			{
				std::cout << "CPU time: " << GetTickCount() - begin << std::endl;
			}

			std::cout << "------------------------------------------" << std::endl;
			begin = GetTickCount();
			(*CPU_PARALLEL_Func)(A, B, C, r, c);
			if (Func_exist == true)
			{
				CPU_PARALLEL_Vec.at(counter) = GetTickCount() - begin;
				std::cout << "CPU PARALLEL time: " << CPU_PARALLEL_Vec.at(counter) << std::endl;
			}
			else
			{
				std::cout << "CPU PARALLEL time: " << GetTickCount() - begin << std::endl;
			}

			std::cout << "------------------------------------------" << std::endl;
			begin = GetTickCount();
			(*GPU_Func)(A, B, C, r, c);
			if (Func_exist == true)
			{
				GPU_Vec.at(counter) = GetTickCount() - begin;
				std::cout << "GPU time: " << GPU_Vec.at(counter) << std::endl;
			}
			else
			{
				std::cout << "GPU time: " << GetTickCount() - begin << std::endl;
			}
			std::cout << "==========================================" << std::endl;

			Func_exist = true;
			counter++;
		}

		ArrangeSizes(CPU_Vec, CPU_PARALLEL_Vec, GPU_Vec);
	}

	void ArrangeSizes(const std::vector<__int64>& CPU_Vec, const std::vector<__int64>& CPU_PARALLEL_Vec, const std::vector<__int64>& GPU_Vec)
	{
		std::string tmp;
		int num = 2;
		for (int i = 0; i < (int)CPU_Vec.size(); i++)
		{
			num = (int)pow(2,i + 5);
			tmp.append(std::to_string(num));
			if ((CPU_Vec.at(i) <= CPU_PARALLEL_Vec.at(i)) && (CPU_Vec.at(i) <= GPU_Vec.at(i)))
			{
				tmp.append(":CPU, ");
			}
			else if ((CPU_PARALLEL_Vec.at(i) < CPU_Vec.at(i)) && (CPU_PARALLEL_Vec.at(i) <= GPU_Vec.at(i)))
			{
				tmp.append(":CPU PARALLEL, ");
			}
			else
			{
				tmp.append(":GPU, ");
			}
		}

		PrintToLogFile(tmp);
	}

	void PrintToLogFile(const std::string& arrangment, const std::string& _file)
	{
		std::ofstream myfile;
		myfile.open(_file.c_str(), std::ios::app);
		time_t end_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		myfile << "Created date:" << time(&end_time) << std::endl;
		myfile << arrangment.c_str() << std::endl;
	}

	void Run(int RowLimit, int ColLimit)
	{
		float* A = new float[RowLimit * ColLimit];
		InitMatrix(A, RowLimit, ColLimit);
		float* B = new float[RowLimit * ColLimit];
		InitMatrix(B, RowLimit, ColLimit);
		float* C = new float[RowLimit * ColLimit];
		InitMatrix(C, RowLimit, ColLimit);

		for (int i = 0; i < 4; i++)
			SingleRun(A, B, C, RowLimit, ColLimit, i);
	}
}