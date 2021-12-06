#include <random>
#include <benchmark/benchmark.h>
#include "cclib.h"

class Benchmark : public benchmark::Fixture
{
public:
	void SetUp(const ::benchmark::State& state)
	{
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_real_distribution<float> dist(0.0, 360.0);
		for (int i = 0; i < TESTNUM; ++i)
		{
			values[i] = dist(mt);
		}
	}

	static constexpr int TESTNUM = 65536;
	float values[TESTNUM];
};

BENCHMARK_DEFINE_F(Benchmark, STD_RSQRT)(benchmark::State& st)
{
	double x = 0.0;
	for (auto _ : st)
	{
		for (int i = 0; i < TESTNUM; ++i)
		{
			benchmark::DoNotOptimize(x += (1.f / sqrtf(values[i])));
			benchmark::ClobberMemory();
		}
	}
}

BENCHMARK_DEFINE_F(Benchmark, CC_RSQRT)(benchmark::State& st)
{
	double x = 0.0;
	for (auto _ : st)
	{
		for (int i = 0; i < TESTNUM; ++i)
		{
			benchmark::DoNotOptimize(x += cc::math::rsqrt(values[i]));
			benchmark::ClobberMemory();
		}
	}
}

BENCHMARK_DEFINE_F(Benchmark, STD_SINCOS)(benchmark::State& st)
{
	double x = 0.0;
	for (auto _ : st)
	{
		for (int i = 0; i < TESTNUM; ++i)
		{
			benchmark::DoNotOptimize(x += ::sinf(values[i]) + ::cosf(values[i]));
			benchmark::ClobberMemory();
		}
	}
}

BENCHMARK_DEFINE_F(Benchmark, CC_SINCOS)(benchmark::State& st)
{
	double x = 0.0;
	for (auto _ : st)
	{
		for (int i = 0; i < TESTNUM; ++i)
		{
			float s, c;
			cc::math::sincosf(values[i], &s, &c);
			benchmark::DoNotOptimize(x += s + c);
			benchmark::ClobberMemory();
		}
	}
}

BENCHMARK_DEFINE_F(Benchmark, STD_ATAN2)(benchmark::State& st)
{
	double x = 0.0;
	for (auto _ : st)
	{
		for (int i = 0; i < TESTNUM; ++i)
		{
			benchmark::DoNotOptimize(x += ::atan2f(values[i], values[i]));
			benchmark::ClobberMemory();
		}
	}
}

BENCHMARK_DEFINE_F(Benchmark, CC_ATAN2)(benchmark::State& st)
{
	double x = 0.0;
	for (auto _ : st)
	{
		for (int i = 0; i < TESTNUM; ++i)
		{
			benchmark::DoNotOptimize(x += cc::math::atan2f(values[i], values[i]));
			benchmark::ClobberMemory();
		}
	}
}

BENCHMARK_REGISTER_F(Benchmark, STD_RSQRT);
BENCHMARK_REGISTER_F(Benchmark, CC_RSQRT);
BENCHMARK_REGISTER_F(Benchmark, STD_SINCOS);
BENCHMARK_REGISTER_F(Benchmark, CC_SINCOS);
BENCHMARK_REGISTER_F(Benchmark, STD_ATAN2);
BENCHMARK_REGISTER_F(Benchmark, CC_ATAN2);

BENCHMARK_MAIN();
