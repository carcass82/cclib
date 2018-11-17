#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include <chrono>
using namespace std::chrono_literals;
using std::chrono::high_resolution_clock;

#include <random>

#include "../cclib.h"

const int TESTNUM = 65536;

void test_atan2(float* values, float* avg_cc, float* avg_std, float* rmse)
{
	float res_cclib[TESTNUM];
	float res_stdlib[TESTNUM];
	
	high_resolution_clock::time_point t0 = high_resolution_clock::now();
	
	for (int i = 0; i < TESTNUM; ++i) { res_cclib[i] = cc::math::fast::atan2f(values[i], values[TESTNUM - i]); }
	
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	
	for (int i = 0; i < TESTNUM; ++i) { res_stdlib[i] = atan2f(values[i], values[TESTNUM - i]); }
	
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	
	*avg_cc = ((float)(t1 - t0).count()) / (TESTNUM);
	*avg_std = ((float)(t2 - t1).count()) / (TESTNUM);
	
	*rmse = .0f;
	for (int i = 0; i < TESTNUM; ++i)
	{
		float diff = res_stdlib[i] - res_cclib[i];
		*rmse += (diff * diff);
	}
	*rmse /= TESTNUM;
	*rmse = sqrtf(*rmse);
}

void test_rcp(float* values, float* avg_cc, float* avg_std, float* rmse)
{
	float res_cclib[TESTNUM];
	float res_stdlib[TESTNUM];
	
	high_resolution_clock::time_point t0 = high_resolution_clock::now();
	
	for (int i = 0; i < TESTNUM; ++i) { res_cclib[i] = cc::math::fast::rcp(values[i]); }
	
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	
	for (int i = 0; i < TESTNUM; ++i) { res_stdlib[i] = 1.f / values[i]; }
	
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	
	*avg_cc = ((float)(t1 - t0).count()) / (TESTNUM);
	*avg_std = ((float)(t2 - t1).count()) / (TESTNUM);
	
	*rmse = .0f;
	for (int i = 0; i < TESTNUM; ++i)
	{
		float diff = res_stdlib[i] - res_cclib[i];
		*rmse += (diff * diff);
	}
	*rmse /= TESTNUM;
	*rmse = sqrtf(*rmse);
}

void test_rsqrt(float* values, float* avg_cc, float* avg_std, float* rmse)
{
	float res_cclib[TESTNUM];
	float res_stdlib[TESTNUM];
	
	high_resolution_clock::time_point t0 = high_resolution_clock::now();
	
	for (int i = 0; i < TESTNUM; ++i) { res_cclib[i] = cc::math::fast::rsqrt(values[i]); }
	
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	
	for (int i = 0; i < TESTNUM; ++i) { res_stdlib[i] = 1.f / sqrtf(values[i]); }
	
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	
	*avg_cc = ((float)(t1 - t0).count()) / (TESTNUM);
	*avg_std = ((float)(t2 - t1).count()) / (TESTNUM);
	
	*rmse = .0f;
	for (int i = 0; i < TESTNUM; ++i)
	{
		float diff = res_stdlib[i] - res_cclib[i];
		*rmse += (diff * diff);
	}
	*rmse /= TESTNUM;
	*rmse = sqrtf(*rmse);
}

void test_sin(float* values, float* avg_cc, float* avg_std, float* rmse)
{
	float res_cclib[TESTNUM];
	float res_stdlib[TESTNUM];
	
	high_resolution_clock::time_point t0 = high_resolution_clock::now();
	
	for (int i = 0; i < TESTNUM; ++i) { res_cclib[i] = cc::math::fast::sinf(values[i]); }
	
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	
	for (int i = 0; i < TESTNUM; ++i) { res_stdlib[i] = sinf(values[i]); }
	
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	
	*avg_cc = ((float)(t1 - t0).count()) / (TESTNUM);
	*avg_std = ((float)(t2 - t1).count()) / (TESTNUM);
	
	*rmse = .0f;
	for (int i = 0; i < TESTNUM; ++i)
	{
		float diff = res_stdlib[i] - res_cclib[i];
		*rmse += (diff * diff);
	}
	*rmse /= TESTNUM;
	*rmse = sqrtf(*rmse);
}

void test_cos(float* values, float* avg_cc, float* avg_std, float* rmse)
{
	float res_cclib[TESTNUM];
	float res_stdlib[TESTNUM];
	
	high_resolution_clock::time_point t0 = high_resolution_clock::now();
	
	for (int i = 0; i < TESTNUM; ++i) { res_cclib[i] = cc::math::fast::cosf(values[i]); }
	
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	
	for (int i = 0; i < TESTNUM; ++i) { res_stdlib[i] = cosf(values[i]); }
	
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	
	*avg_cc = ((float)(t1 - t0).count()) / (TESTNUM);
	*avg_std = ((float)(t2 - t1).count()) / (TESTNUM);
	
	*rmse = .0f;
	for (int i = 0; i < TESTNUM; ++i)
	{
		float diff = res_stdlib[i] - res_cclib[i];
		*rmse += (diff * diff);
	}
	*rmse /= TESTNUM;
	*rmse = sqrtf(*rmse);
}

int main()
{
	float values[TESTNUM];
	
	std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 100.0);
	for (int i = 0; i < TESTNUM; ++i)
	{
		values[i] = dist(mt);
	}
	
	float avg_cclib = .0f;
	float avg_stdlib = .0f;
	float rmse = .0f;
	
	test_atan2(values, &avg_cclib, &avg_stdlib, &rmse);
	printf("atan2f(y, x): cclib: %02.2f / stdlib: %02.2f (speedup: %+d%% RMSE: %.3f)\n", avg_cclib, avg_stdlib, int(((avg_stdlib - avg_cclib) * 100.f / avg_stdlib)), rmse);
	
	test_rcp(values, &avg_cclib, &avg_stdlib, &rmse);
	printf("rcp(x):       cclib: %02.2f / stdlib: %02.2f (speedup: %+d%% RMSE: %.3f)\n", avg_cclib, avg_stdlib, int(((avg_stdlib - avg_cclib) * 100.f / avg_stdlib)), rmse);
	
	test_rsqrt(values, &avg_cclib, &avg_stdlib, &rmse);
	printf("rsqrtf(x):    cclib: %02.2f / stdlib: %02.2f (speedup: %+d%% RMSE: %.3f)\n", avg_cclib, avg_stdlib, int(((avg_stdlib - avg_cclib) * 100.f / avg_stdlib)), rmse);
	
	test_sin(values, &avg_cclib, &avg_stdlib, &rmse);
	printf("sinf(x):      cclib: %02.2f / stdlib: %02.2f (speedup: %+d%% RMSE: %.3f)\n", avg_cclib, avg_stdlib, int(((avg_stdlib - avg_cclib) * 100.f / avg_stdlib)), rmse);
	
	test_cos(values, &avg_cclib, &avg_stdlib, &rmse);
	printf("cosf(x):      cclib: %02.2f / stdlib: %02.2f (speedup: %+d%% RMSE: %.3f)\n", avg_cclib, avg_stdlib, int(((avg_stdlib - avg_cclib) * 100.f / avg_stdlib)), rmse);	
	
	return 0;
}
