#ifndef _PERFORMANCE_H_
#define _PERFORMANCE_H_

#include "IncluderLib.h"

namespace GPerformance {
	template <class Function>
	__int64 time_call(Function&& f())
	{
		__int64 begin = GetTickCount();
		f();
		return GetTickCount() - begin;
	}

	void InitMatrix(float* A, int r_s, int c_s);

	void GPU_mul(float* A, float* B, float* C, int r_s, int c_s);
	void CPU_PARALLEL_mul(float* A, float* B, float* C, int r_s, int c_s);
	void CPU_mul(float* A, float* B, float* C, int r_s, int c_s);

	void GPU_plus_Val(float* A, float* B, float* C, int r_s, int c_s);
	void CPU_PARALLEL_plus_Val(float* A, float* B, float* C, int r_s, int c_s);
	void CPU_plus_Val(float* A, float* B, float* C, int r_s, int c_s);

	void GPU_ReLU(float* A, float* B, float* C, int r_s, int c_s);
	void CPU_PARALLEL_ReLU(float* A, float* B, float* C, int r_s, int c_s);
	void CPU_ReLU(float* A, float* B, float* C, int r_s, int c_s);

	void GPU_conv(float* A, float* B, float* C, int r_s, int c_s);
	void CPU_PARALLEL_conv(float* A, float* B, float* C, int r_s, int c_s);
	void CPU_conv(float* A, float* B, float* C, int r_s, int c_s);

	void SingleRun(float* A, float* B, float* C, int r_s, int c_s, int index);
	void ArrangeSizes(const std::vector<__int64>& CPU_Vec, const std::vector<__int64>& CPU_PARALLEL_Vec, const std::vector<__int64>& GPU_Vec);
	void PrintTime(void(*CPU_Func)(float*, float*, float*, int, int), void(*CPU_PARALLEL_Func)(float*, float*, float*, int, int),
		void(*GPU_Func)(float*, float*, float*, int, int), float* A, float* B, float* C, int r_s, int c_s);
	void PrintToLogFile(const std::string& arrangment, const std::string& _file = "Performance.txt");
	void Run(int RowLimit = 2048, int ColLimit = 2048);

}
#endif // !_PERFORMANCE_H_

