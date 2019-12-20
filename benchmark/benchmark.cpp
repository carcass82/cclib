#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include <chrono>
using namespace std::chrono_literals;
using std::chrono::high_resolution_clock;

#include <random>

#if defined(CHECK_CORRECTNESS)
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#endif

#include "../cclib.h"

const int TESTNUM = 65536;

void test_atan2(float* values, float* avg_cc, float* avg_std, float* rmse)
{
	float res_cclib[TESTNUM];
	float res_stdlib[TESTNUM];
	
	high_resolution_clock::time_point t0 = high_resolution_clock::now();
	
	for (int i = 0; i < TESTNUM; ++i) { res_cclib[i] = cc::math::atan2f(values[i], values[TESTNUM - i]); }
	
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
	
	for (int i = 0; i < TESTNUM; ++i) { res_cclib[i] = cc::math::rcp(values[i]); }
	
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
	
	for (int i = 0; i < TESTNUM; ++i) { res_cclib[i] = cc::math::rsqrt(values[i]); }
	
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
	
	for (int i = 0; i < TESTNUM; ++i) { res_cclib[i] = cc::math::sinf(values[i]); }
	
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
	
	for (int i = 0; i < TESTNUM; ++i) { res_cclib[i] = cc::math::cosf(values[i]); }
	
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
	printf("atan2f(y, x): cclib: %05.2fms - stdlib: %05.2fms (speedup: %+d%% RMSE: %.3f)\n", avg_cclib, avg_stdlib, int(((avg_stdlib - avg_cclib) * 100.f / avg_stdlib)), rmse);
	
	test_rcp(values, &avg_cclib, &avg_stdlib, &rmse);
	printf("rcp(x):       cclib: %05.2fms - stdlib: %05.2fms (speedup: %+d%% RMSE: %.3f)\n", avg_cclib, avg_stdlib, int(((avg_stdlib - avg_cclib) * 100.f / avg_stdlib)), rmse);
	
	test_rsqrt(values, &avg_cclib, &avg_stdlib, &rmse);
	printf("rsqrtf(x):    cclib: %05.2fms - stdlib: %05.2fms (speedup: %+d%% RMSE: %.3f)\n", avg_cclib, avg_stdlib, int(((avg_stdlib - avg_cclib) * 100.f / avg_stdlib)), rmse);
	
	test_sin(values, &avg_cclib, &avg_stdlib, &rmse);
	printf("sinf(x):      cclib: %05.2fms - stdlib: %05.2fms (speedup: %+d%% RMSE: %.3f)\n", avg_cclib, avg_stdlib, int(((avg_stdlib - avg_cclib) * 100.f / avg_stdlib)), rmse);
	
	test_cos(values, &avg_cclib, &avg_stdlib, &rmse);
	printf("cosf(x):      cclib: %05.2fms - stdlib: %05.2fms (speedup: %+d%% RMSE: %.3f)\n", avg_cclib, avg_stdlib, int(((avg_stdlib - avg_cclib) * 100.f / avg_stdlib)), rmse);	
	
#if defined(CHECK_CORRECTNESS)
	float eye[] = { 2.f, 5.f, 10.f };
	float center[] = { .0f, .0f, .0f };
	float up[] = { .0f, 1.f, .0f };
	float fovy = 1.05f;
	float aspect = 1.33f;
	float near = 0.1f;
	float far = 1000.f;
	float vpos[] = { 3.f, 2.f, 1.f, 1.f };

	glm::mat4 glm_V = glm::lookAt(glm::vec3(eye[0], eye[1], eye[2]),
	                              glm::vec3(center[0], center[1], center[2]),
								  glm::vec3(up[0], up[1], up[2]));
	glm::mat4 glm_P = glm::perspective(fovy, aspect, near, far);
	glm::mat4 glm_P_V = glm_P * glm_V;
	glm::vec4 glm_pos = glm_P * glm_V * glm::vec4(vpos[0], vpos[1], vpos[2], vpos[3]);

	cc::math::mat4 cc_V = cc::math::lookAt(cc::math::vec3(eye[0], eye[1], eye[2]),
	                                       cc::math::vec3(center[0], center[1], center[2]),
								           cc::math::vec3(up[0], up[1], up[2]));
	cc::math::mat4 cc_P = cc::math::perspective(fovy, aspect, near, far);
	cc::math::mat4 cc_P_V = cc_P * cc_V;
	cc::math::vec4 cc_pos = cc_P_V * cc::math::vec4(vpos[0], vpos[1], vpos[2], vpos[3]);

	printf("\nView: GLM                            CC\n");
	for (int i = 0; i < 4; ++i)
		printf("      [%+06.2f %+06.2f %+06.2f %+06.2f]  [%+06.2f %+06.2f %+06.2f %+06.2f]\n",
			glm_V[i].x, glm_V[i].y, glm_V[i].z, glm_V[i].w,
			cc_V[i].x, cc_V[i].y, cc_V[i].z, cc_V[i].w);
	printf("\n");
	
	printf("Projection: GLM                            CC\n");
	for (int i = 0; i < 4; ++i)
		printf("            [%+06.2f %+06.2f %+06.2f %+06.2f]  [%+06.2f %+06.2f %+06.2f %+06.2f]\n",
			glm_P[i].x, glm_P[i].y, glm_P[i].z, glm_P[i].w,
			cc_P[i].x, cc_P[i].y, cc_P[i].z, cc_P[i].w);
	printf("\n");
	
	printf("mat4 * mat4: GLM                            CC\n");
	for (int i = 0; i < 4; ++i)
		printf("             [%+06.2f %+06.2f %+06.2f %+06.2f]  [%+06.2f %+06.2f %+06.2f %+06.2f]\n",
			glm_P_V[i].x, glm_P_V[i].y, glm_P_V[i].z, glm_P_V[i].w,
			cc_P_V[i].x, cc_P_V[i].y, cc_P_V[i].z, cc_P_V[i].w);
	printf("\n");

	printf("mat4 * vec4: GLM                            CC\n");
	printf("             [%+06.2f %+06.2f %+06.2f %+06.2f]  [%+06.2f %+06.2f %+06.2f %+06.2f]\n\n",
		glm_pos.x, glm_pos.y, glm_pos.z, glm_pos.w,
		cc_pos.x, cc_pos.y, cc_pos.z, cc_pos.w);

	printf("inverse(mat3): GLM                     CC\n");
	for (int i = 0; i < 3; ++i)
		printf("               [%+06.2f %+06.2f %+06.2f]  [%+06.2f %+06.2f %+06.2f]\n",
			glm::inverse(glm::mat3(glm_P_V))[i].x, glm::inverse(glm::mat3(glm_P_V))[i].y, glm::inverse(glm::mat3(glm_P_V))[i].z,
			cc::math::inverse(cc::math::mat3(cc_P_V))[i].x, cc::math::inverse(cc::math::mat3(cc_P_V))[i].y, cc::math::inverse(cc::math::mat3(cc_P_V))[i].z);
	printf("\n");
    
    printf("inverse(mat4): GLM                            CC\n");
	for (int i = 0; i < 4; ++i)
		printf("               [%+06.2f %+06.2f %+06.2f %+06.2f]  [%+06.2f %+06.2f %+06.2f %+06.2f]\n",
			glm::inverse(glm::mat4(glm_P_V))[i].x, glm::inverse(glm::mat4(glm_P_V))[i].y, glm::inverse(glm::mat4(glm_P_V))[i].z, glm::inverse(glm::mat4(glm_P_V))[i].w,
			cc::math::inverse(cc::math::mat4(cc_P_V))[i].x, cc::math::inverse(cc::math::mat4(cc_P_V))[i].y, cc::math::inverse(cc::math::mat4(cc_P_V))[i].z, cc::math::inverse(cc::math::mat4(cc_P_V))[i].w);
	printf("\n");
#endif

	return 0;
}
