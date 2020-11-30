/*
 * CCLib
 *
 * collection of utils i use in most projects.
 * maybe it will evolve in a framework, maybe not
 *
 * (c) 2018 Carlo Casta <carlo.casta at gmail.com>
 */
#pragma once

#include <cstdint>
#include <utility>
#include <cmath>

#if defined(_MSC_VER)
 #include <intrin.h>
#else
 #include <x86intrin.h>
#endif

#if defined(__CUDACC__)
 #define CUDA_DEVICE_CALL __device__
 #define CUDA_CALL __host__ __device__
#else
 #define CUDA_DEVICE_CALL
 #define CUDA_CALL
#endif

#if !defined(CC_CONSTEXPR_MATH)
 #define CC_CONSTEXPR
#else
 #define CC_CONSTEXPR constexpr
#endif

namespace cc
{
    template<typename T, size_t N>
    CUDA_CALL constexpr inline uint32_t array_size(const T(&)[N]) { return N; }

    template<typename T>
    CUDA_CALL constexpr inline void swap(T& a, T& b) { T tmp = std::move(a); a = std::move(b); b = std::move(tmp); }

namespace math
{
    //
    // math constants
    //
    constexpr float PI = 3.1415926535897932f;
    constexpr float PI_2 = 1.57079632679f;
    constexpr float EPS = 1.e-8f;

    template<typename T>
    CUDA_CALL constexpr inline T min(const T& a, const T& b) { return !(b < a) ? a : b; }

    template<typename T>
    CUDA_CALL constexpr inline T max(const T& a, const T& b) { return (a < b) ? b : a; }

    template<typename T>
    CUDA_CALL constexpr inline T clamp(const T& a, const T& lower, const T& upper) { return min(max(a, lower), upper); }

    template<typename T>
    CUDA_CALL constexpr inline T saturate(const T& a) { return clamp(a, T(0), T(1)); }

    template<typename T>
    CUDA_CALL constexpr inline T sign(const T& x) { return T((x > T(0)) - (x < T(0))); }

    template<typename T>
    CUDA_CALL constexpr inline T abs(const T& a) { return (a < T(0)) ? -a : a; }

    template<typename T>
    CUDA_CALL constexpr inline T lerp(const T v0, const T v1, float t) { return v0 + t * (v1 - v0); }

    template<typename T>
    CUDA_CALL constexpr inline bool are_equal(const T a, const T b) { return a == b; }

    // float comparison, see http://realtimecollisiondetection.net/pubs/Tolerances/
    template<>
    CUDA_CALL constexpr inline bool are_equal(const float a, const float b) { return abs(a - b) <= EPS * max(max(1.f, abs(a)), abs(b)); }

    CUDA_CALL CC_CONSTEXPR inline float pow(float x, float y)
    {
        return ::powf(x, y);
    }

    CUDA_CALL CC_CONSTEXPR inline float sqrtf(float x)
    {
        return ::sqrtf(x);
    }

    CUDA_CALL constexpr inline float rcp(float x)
    {
		return 1.f / x;
    }
        
    CUDA_CALL CC_CONSTEXPR inline float rsqrt(float x)
    {
		return rcp(sqrtf(x));
    }

    CUDA_CALL CC_CONSTEXPR inline float sinf(float x)
    {
        return ::sinf(x);
    }

    CUDA_CALL CC_CONSTEXPR inline float cosf(float x)
    {
        return ::cosf(x);
    }

    CUDA_CALL CC_CONSTEXPR inline void sincosf(float x, float* s, float* c)
    {
        *s = sinf(x);
        *c = cosf(x);
    }

    CUDA_CALL CC_CONSTEXPR inline float tanf(float x)
    {
        return ::tanf(x);
    }

    CUDA_CALL CC_CONSTEXPR inline float atan2f(float y, float x)
    {
        return ::atan2f(y, x);
    }

    CUDA_CALL CC_CONSTEXPR inline float cotf(float x)
    {
        return rcp(tanf(x));
    }

    //
    // conversion utils
    //
    CUDA_CALL constexpr inline float radians(float deg)                  { return deg * PI / 180.0f; }

    CUDA_CALL constexpr inline float degrees(float rad)                  { return rad * 180.0f / PI; }

    //
    // useful types
    //
    struct vec2
    {
        union {
            float v[2];
            struct { float x, y; }; struct { float xy[2]; };
            struct { float s, t; }; struct { float st[2]; };
            struct { float w, h; }; struct { float wh[2]; };
        };


        CUDA_CALL constexpr inline float& operator[](size_t i)             { return v[i]; }

        CUDA_CALL constexpr inline const float& operator[](size_t i) const { return v[i]; }


        CUDA_CALL constexpr inline vec2() noexcept                     : v{} {}

        CUDA_CALL constexpr inline vec2(float _v) noexcept             : v{_v, _v} {}

        CUDA_CALL constexpr inline vec2(float _v1, float _v2) noexcept : v{ _v1, _v2 } {}

        CUDA_CALL constexpr inline vec2(const float _v[2]) noexcept    : v{ _v[0], _v[1] } {}
    };

    struct vec3
    {
        union {
            float v[3];
            struct { float x, y, z; }; struct { float xy[2], z0; }; struct { float x0, yz[2]; }; struct { float xyz[3]; };
            struct { float r, g, b; }; struct { float rg[2], b0; }; struct { float r0, gb[2]; }; struct { float rgb[3]; };
        };


        CUDA_CALL constexpr inline float& operator[](size_t i)             { return v[i]; }

        CUDA_CALL constexpr inline const float& operator[](size_t i) const { return v[i]; }


        CUDA_CALL constexpr inline vec3() noexcept                                : v{} {}

        CUDA_CALL constexpr inline vec3(float _v) noexcept                        : v{_v, _v, _v} {}

        CUDA_CALL constexpr inline vec3(float _v1, float _v2, float _v3) noexcept : v{ _v1, _v2, _v3 } {}

        CUDA_CALL constexpr inline vec3(const float _v[3]) noexcept               : v{ _v[0], _v[1], _v[2] } {}

        CUDA_CALL constexpr inline vec3(const vec2& _vec, float _v) noexcept      : v{ _vec.x, _vec.y, _v } {}

        CUDA_CALL constexpr inline vec3(float _v, const vec2& _vec) noexcept      : v{ _v, _vec.x, _vec.y } {}
    };

    struct alignas(16) vec4
    {
        union {
            float v[4];
            struct { float x, y, z, w; }; struct { float xy[2], z0, w0; }; struct { float x0, y0, zw[2]; }; struct { float xyz[3], w1; }; struct { float x1, yzw[3]; }; struct { float xyzw[4]; };
            struct { float r, g, b, a; }; struct { float rg[2], b0, a0; }; struct { float r0, g0, ba[2]; }; struct { float rgb[3], a1; }; struct { float r1, gba[3]; }; struct { float rgba[4]; };
        };

        CUDA_CALL constexpr inline float& operator[](size_t i)             { return v[i]; }

        CUDA_CALL constexpr inline const float& operator[](size_t i) const { return v[i]; }


        CUDA_CALL constexpr inline vec4() noexcept                                           : v{} {}

        CUDA_CALL constexpr inline vec4(float _v) noexcept                                   : v{_v, _v, _v, _v} {}

        CUDA_CALL constexpr inline vec4(float _v1, float _v2, float _v3, float _v4) noexcept : v{ _v1, _v2, _v3, _v4 } {}

        CUDA_CALL constexpr inline vec4(const float _v[4]) noexcept                          : v{ _v[0], _v[1], _v[2], _v[3] } {}

        CUDA_CALL constexpr inline vec4(const vec2& _vec1, const vec2& _vec2) noexcept       : v{ _vec1.x, _vec1.y, _vec2.x, _vec2.y } {}

		CUDA_CALL constexpr inline vec4(const vec3& _vec, float _v) noexcept                 : v{_vec.x, _vec.y, _vec.z, _v} {}

        CUDA_CALL constexpr inline vec4(float _v, const vec3& _vec) noexcept                 : v{ _v, _vec.x, _vec.y, _vec.z } {}
    };

    struct alignas(64) mat4
    {
        union {
            vec4 m[4];
            struct {
                float _m00, _m10, _m20, _m30;
                float _m01, _m11, _m21, _m31;
                float _m02, _m12, _m22, _m32;
                float _m03, _m13, _m23, _m33;
            };
        };


        CUDA_CALL constexpr vec4& operator[](size_t i)             { return m[i]; }

        CUDA_CALL constexpr const vec4& operator[](size_t i) const { return m[i]; }


        CUDA_CALL constexpr inline mat4() noexcept                                                                        : m{} {}

        CUDA_CALL constexpr inline explicit mat4(float _i) noexcept                                                       : m{} { _m00 = _m11 = _m22 = _m33 = _i; }

        CUDA_CALL constexpr inline explicit mat4(const vec4& v0, const vec4& v1, const vec4& v2, const vec4& v3) noexcept : m{ v0, v1, v2, v3 } {}
    };

    struct mat3
    {
        union {
            vec3 m[3];
            struct {
                float _m00, _m10, _m20;
                float _m01, _m11, _m21;
                float _m02, _m12, _m22;
            };
        };


        CUDA_CALL constexpr vec3& operator[](size_t i)             { return m[i]; }

        CUDA_CALL constexpr const vec3& operator[](size_t i) const { return m[i]; }


        CUDA_CALL constexpr inline mat3() noexcept : m{} {}

        CUDA_CALL constexpr inline explicit mat3(float _i) noexcept                                       : m{} { _m00 = _m11 = _m22 = _i; }

        CUDA_CALL constexpr inline explicit mat3(const vec3& v0, const vec3& v1, const vec3& v2) noexcept : m{ v0, v1, v2 } {}

        CUDA_CALL constexpr inline explicit mat3(const mat4& _m) noexcept                                 : m{ _m[0].xyz, _m[1].xyz, _m[2].xyz } {}
    };

    //
    // compatibility with GLM
    //
    CUDA_CALL constexpr inline const float* value_ptr(const vec2& v)                   { return &(v.v[0]); }
    CUDA_CALL constexpr inline const float* value_ptr(const vec3& v)                   { return &(v.v[0]); }
    CUDA_CALL constexpr inline const float* value_ptr(const vec4& v)                   { return &(v.v[0]); }
    CUDA_CALL constexpr inline const float* value_ptr(const mat3& m)                   { return value_ptr(m.m[0]); }
    CUDA_CALL constexpr inline const float* value_ptr(const mat4& m)                   { return value_ptr(m.m[0]); }

    //
    // operators
    //
    CUDA_CALL constexpr inline vec2 operator+(const vec2& a, float b)                  { return vec2{ a.x + b, a.y + b }; }
    CUDA_CALL constexpr inline vec2 operator+(float b, const vec2& a)                  { return vec2{ a.x + b, a.y + b }; }
    CUDA_CALL constexpr inline vec2 operator+(const vec2& a, const vec2& b)            { return vec2{ a.x + b.x, a.y + b.y }; }
    CUDA_CALL constexpr inline vec2 operator-(const vec2& a)                           { return vec2{ -a.x, -a.y }; }
    CUDA_CALL constexpr inline vec2 operator-(const vec2& a, float b)                  { return vec2{ a.x - b, a.y - b }; }
    CUDA_CALL constexpr inline vec2 operator-(float b, const vec2& a)                  { return vec2{ a.x - b, a.y - b }; }
    CUDA_CALL constexpr inline vec2 operator-(const vec2& a, const vec2& b)            { return vec2{ a.x - b.x, a.y - b.y }; }
    CUDA_CALL constexpr inline vec2 operator*(const vec2& a, float b)                  { return vec2{ a.x * b, a.y * b }; }
    CUDA_CALL constexpr inline vec2 operator*(float b, const vec2& a)                  { return vec2{ a.x * b, a.y * b }; }
    CUDA_CALL constexpr inline vec2 operator*(const vec2& a, const vec2& b)            { return vec2{ a.x * b.x, a.y * b.y }; }
    CUDA_CALL constexpr inline vec2 operator/(const vec2& a, float b)                  { return vec2{ a.x / b, a.y / b }; }
    CUDA_CALL constexpr inline vec2 operator/(float a, const vec2& b)                  { return vec2{ a / b.x, a / b.y }; }
    CUDA_CALL constexpr inline vec2 operator/(const vec2& a, const vec2& b)            { return vec2{ a.x / b.x, a.y / b.y }; }
    CUDA_CALL constexpr inline vec2& operator+=(vec2& a, float b)                      { a.x += b; a.y += b; return a; }
    CUDA_CALL constexpr inline vec2& operator-=(vec2& a, float b)                      { a.x -= b; a.y -= b; return a; }
    CUDA_CALL constexpr inline vec2& operator*=(vec2& a, float b)                      { a.x *= b; a.y *= b; return a; }
    CUDA_CALL constexpr inline vec2& operator/=(vec2& a, float b)                      { a.x /= b; a.y /= b; return a; }
    CUDA_CALL constexpr inline bool operator==(const vec2& a, const vec2& b)           { return are_equal(a.x, b.x) && are_equal(a.y, b.y); }
    CUDA_CALL constexpr inline bool operator!=(const vec2& a, const vec2& b)           { return !(a == b); }
    CUDA_CALL constexpr inline vec2 pmax(const vec2& a, const vec2& b)                 { return vec2{ max(a.x, b.x), max(a.y, b.y) }; }
    CUDA_CALL constexpr inline vec2 pmin(const vec2& a, const vec2& b)                 { return vec2{ min(a.x, b.x), min(a.y, b.y) }; }

    CUDA_CALL constexpr inline vec3 operator+(const vec3& a, float b)                  { return vec3{ a.x + b, a.y + b, a.z + b }; }
    CUDA_CALL constexpr inline vec3 operator+(float b, const vec3& a)                  { return vec3{ a.x + b, a.y + b, a.z + b }; }
    CUDA_CALL constexpr inline vec3 operator+(const vec3& a, const vec3& b)            { return vec3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
    CUDA_CALL constexpr inline vec3 operator-(const vec3& a)                           { return vec3{ -a.x, -a.y, -a.z }; }
    CUDA_CALL constexpr inline vec3 operator-(const vec3& a, float b)                  { return vec3{ a.x - b, a.y - b, a.z - b }; }
    CUDA_CALL constexpr inline vec3 operator-(float b, const vec3& a)                  { return vec3{ a.x - b, a.y - b, a.z - b }; }
    CUDA_CALL constexpr inline vec3 operator-(const vec3& a, const vec3& b)            { return vec3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
    CUDA_CALL constexpr inline vec3 operator*(const vec3& a, float b)                  { return vec3{ a.x * b, a.y * b, a.z * b }; }
    CUDA_CALL constexpr inline vec3 operator*(float b, const vec3& a)                  { return vec3{ a.x * b, a.y * b, a.z * b }; }
    CUDA_CALL constexpr inline vec3 operator*(const vec3& a, const vec3& b)            { return vec3{ a.x * b.x, a.y * b.y, a.z * b.z }; }
    CUDA_CALL constexpr inline vec3 operator/(const vec3& a, float b)                  { return vec3{ a.x / b, a.y / b, a.z / b }; }
    CUDA_CALL constexpr inline vec3 operator/(float a, const vec3& b)                  { return vec3{ a / b.x, a / b.y, a / b.z }; }
    CUDA_CALL constexpr inline vec3 operator/(const vec3& a, const vec3& b)            { return vec3{ a.x / b.x, a.y / b.y, a.z / b.z }; }
    CUDA_CALL constexpr inline vec3& operator+=(vec3& a, const vec3& b)                { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
    CUDA_CALL constexpr inline vec3& operator-=(vec3& a, const vec3& b)                { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
    CUDA_CALL constexpr inline vec3& operator*=(vec3& a, const vec3& b)                { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
    CUDA_CALL constexpr inline vec3& operator/=(vec3& a, const vec3& b)                { a.x /= b.x; a.y /= b.y; a.z /= b.z; return a; }
    CUDA_CALL constexpr inline vec3& operator+=(vec3& a, float b)                      { a.x += b; a.y += b; a.z += b; return a; }
    CUDA_CALL constexpr inline vec3& operator-=(vec3& a, float b)                      { a.x -= b; a.y -= b; a.z -= b; return a; }
    CUDA_CALL constexpr inline vec3& operator*=(vec3& a, float b)                      { a.x *= b; a.y *= b; a.z *= b; return a; }
    CUDA_CALL constexpr inline vec3& operator/=(vec3& a, float b)                      { a.x /= b; a.y /= b; a.z /= b; return a; }
    CUDA_CALL constexpr inline bool operator==(const vec3& a, const vec3& b)           { return are_equal(a.x, b.x) && are_equal(a.y, b.y) && are_equal(a.z, b.z); }
    CUDA_CALL constexpr inline bool operator!=(const vec3& a, const vec3& b)           { return !(a == b); }
    CUDA_CALL constexpr inline vec3 pmax(const vec3& a, const vec3& b)                 { return vec3{ max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) }; }
    CUDA_CALL constexpr inline vec3 pmin(const vec3& a, const vec3& b)                 { return vec3{ min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) }; }

    CUDA_CALL constexpr inline vec4 operator+(const vec4& a, float b)                  { return vec4{ a.x + b, a.y + b, a.z + b, a.w + b }; }
    CUDA_CALL constexpr inline vec4 operator+(float b, const vec4& a)                  { return vec4{ a.x + b, a.y + b, a.z + b, a.w + b }; }
    CUDA_CALL constexpr inline vec4 operator+(const vec4& a, const vec4& b)            { return vec4{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }
    CUDA_CALL constexpr inline vec4 operator-(const vec4& a)                           { return vec4{ -a.x, -a.y, -a.z, -a.w }; }
    CUDA_CALL constexpr inline vec4 operator-(const vec4& a, float b)                  { return vec4{ a.x - b, a.y - b, a.z - b, a.w - b }; }
    CUDA_CALL constexpr inline vec4 operator-(float b, const vec4& a)                  { return vec4{ a.x - b, a.y - b, a.z - b, a.w - b }; }
    CUDA_CALL constexpr inline vec4 operator-(const vec4& a, const vec4& b)            { return vec4{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }
    CUDA_CALL constexpr inline vec4 operator*(const vec4& a, float b)                  { return vec4{ a.x * b, a.y * b, a.z * b, a.w * b }; }
    CUDA_CALL constexpr inline vec4 operator*(float b, const vec4& a)                  { return vec4{ a.x * b, a.y * b, a.z * b, a.w * b }; }
    CUDA_CALL constexpr inline vec4 operator*(const vec4& a, const vec4& b)            { return vec4{ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w }; }
    CUDA_CALL constexpr inline vec4 operator/(const vec4& a, float b)                  { return vec4{ a.x / b, a.y / b, a.z / b, a.w / b }; }
    CUDA_CALL constexpr inline vec4 operator/(float a, const vec4& b)                  { return vec4{ a / b.x, a / b.y, a / b.z, a / b.w }; }
    CUDA_CALL constexpr inline vec4 operator/(const vec4& a, const vec4& b)            { return vec4{ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w }; }
    CUDA_CALL constexpr inline vec4& operator+=(vec4& a, const vec4& b)                { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
    CUDA_CALL constexpr inline vec4& operator-=(vec4& a, const vec4& b)                { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
    CUDA_CALL constexpr inline vec4& operator*=(vec4& a, const vec4& b)                { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }
    CUDA_CALL constexpr inline vec4& operator/=(vec4& a, const vec4& b)                { a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w; return a; }
    CUDA_CALL constexpr inline vec4& operator+=(vec4& a, float b)                      { a.x += b; a.y += b; a.z += b; a.w += b; return a; }
    CUDA_CALL constexpr inline vec4& operator-=(vec4& a, float b)                      { a.x -= b; a.y -= b; a.z -= b; a.w -= b; return a; }
    CUDA_CALL constexpr inline vec4& operator*=(vec4& a, float b)                      { a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a; }
    CUDA_CALL constexpr inline vec4& operator/=(vec4& a, float b)                      { a.x /= b; a.y /= b; a.z /= b; a.w /= b; return a; }
    CUDA_CALL constexpr inline bool operator==(const vec4& a, const vec4& b)           { return are_equal(a.x, b.x) && are_equal(a.y, b.y) && are_equal(a.z, b.z) && are_equal(a.w, b.w); }
    CUDA_CALL constexpr inline bool operator!=(const vec4& a, const vec4& b)           { return !(a == b); }
    CUDA_CALL constexpr inline vec4 pmax(const vec4& a, const vec4& b)                 { return vec4{ max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w) }; }
    CUDA_CALL constexpr inline vec4 pmin(const vec4& a, const vec4& b)                 { return vec4{ min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w) }; }

    CUDA_CALL constexpr inline bool operator==(const mat3& a, const mat3& b)           { return a[0] == b[0] && a[1] == b[1] && a[2] == b[2]; }
    CUDA_CALL constexpr inline bool operator!=(const mat3& a, const mat3& b)           { return !(a == b); }
    CUDA_CALL constexpr inline bool operator==(const mat4& a, const mat4& b)           { return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3]; }
    CUDA_CALL constexpr inline bool operator!=(const mat4& a, const mat4& b)           { return !(a == b); }

    CUDA_CALL constexpr inline mat3 operator*(const mat3& a, const mat3& b)
    {
        return mat3
        {
            { a[0] * b[0].x + a[1] * b[0].y + a[2] * b[0].z },
            { a[0] * b[1].x + a[1] * b[1].y + a[2] * b[1].z },
            { a[0] * b[2].x + a[1] * b[2].y + a[2] * b[2].z }
        };
    }

    CUDA_CALL constexpr inline vec3 operator*(const mat3& a, const vec3& b)
    {
        return vec3
        {
            b.x * a[0].x + b.y * a[1].x + b.z * a[2].x,
            b.x * a[0].y + b.y * a[1].y + b.z * a[2].y,
            b.x * a[0].z + b.y * a[1].z + b.z * a[2].z
        };
    }

    CUDA_CALL constexpr inline vec3 operator*(const vec3& a, const mat3& b)
    {
        return vec3
        {
            b[0].x * a.x + b[0].y * a.y + b[0].z * a.z,
            b[1].x * a.x + b[1].y * a.y + b[1].z * a.z,
            b[2].x * a.x + b[2].y * a.y + b[2].z * a.z
        };
    }

    CUDA_CALL constexpr inline mat4 operator*(const mat4& a, const mat4& b)
    {
        return mat4
        {
            { a[0] * b[0].x + a[1] * b[0].y + a[2] * b[0].z + a[3] * b[0].w },
            { a[0] * b[1].x + a[1] * b[1].y + a[2] * b[1].z + a[3] * b[1].w },
            { a[0] * b[2].x + a[1] * b[2].y + a[2] * b[2].z + a[3] * b[2].w },
            { a[0] * b[3].x + a[1] * b[3].y + a[2] * b[3].z + a[3] * b[3].w }
        };
    }

    CUDA_CALL constexpr inline vec4 operator*(const mat4& a, const vec4& b)
    {
        return vec4
        {
            b.x * a[0].x + b.y * a[1].x + b.z * a[2].x + b.w * a[3].x,
            b.x * a[0].y + b.y * a[1].y + b.z * a[2].y + b.w * a[3].y,
            b.x * a[0].z + b.y * a[1].z + b.z * a[2].z + b.w * a[3].z,
            b.x * a[0].w + b.y * a[1].w + b.z * a[2].w + b.w * a[3].w
        };
    }

    CUDA_CALL constexpr inline vec4 operator*(const vec4& a, const mat4& b)
	{
		return vec4
		{
			b[0].x * a.x + b[0].y * a.y + b[0].z * a.z + b[0].w * a.w,
			b[1].x * a.x + b[1].y * a.y + b[1].z * a.z + b[1].w * a.w,
			b[2].x * a.x + b[2].y * a.y + b[2].z * a.z + b[2].w * a.w,
			b[3].x * a.x + b[3].y * a.y + b[3].z * a.z + b[3].w * a.w
		};
	}

    //
    // useful specializations
    //
    template<>
    CUDA_CALL constexpr inline vec2 clamp(const vec2& a, const vec2& lower, const vec2& upper)
    {
        return { clamp(a.x, lower.x, upper.x), clamp(a.y, lower.y, upper.y) };
    }

    template<>
    CUDA_CALL constexpr inline vec2 saturate(const vec2& a)
    {
        return { saturate(a.x), saturate(a.y) };
    }

    template<>
    CUDA_CALL constexpr inline vec3 clamp(const vec3& a, const vec3& lower, const vec3& upper)
    {
        return { clamp(a.x, lower.x, upper.x), clamp(a.y, lower.y, upper.y), clamp(a.z, lower.z, upper.z) };
    }

    template<>
    CUDA_CALL constexpr inline vec3 saturate(const vec3& a)
    {
        return { saturate(a.x), saturate(a.y), saturate(a.z) };
    }

    template<>
    CUDA_CALL constexpr inline vec4 clamp(const vec4& a, const vec4& lower, const vec4& upper)
    {
        return { clamp(a.x, lower.x, upper.x), clamp(a.y, lower.y, upper.y), clamp(a.z, lower.z, upper.z), clamp(a.w, lower.w, upper.w) };
    }

    template<>
    CUDA_CALL constexpr inline vec4 saturate(const vec4& a)
    {
        return { saturate(a.x), saturate(a.y), saturate(a.z), saturate(a.w) };
    }
    

    //
    // trig functions
    //
    CUDA_CALL constexpr inline float dot(const vec3& a, const vec3& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    CUDA_CALL constexpr inline float length2(const vec3& a)
    {
        return dot(a, a);
    }

    CUDA_CALL CC_CONSTEXPR inline float length(const vec3& a)
    {
        return sqrtf(length2(a));
    }

    CUDA_CALL constexpr inline float distance2(const vec3& a, const vec3& b)
    {
        return (a.x - b.x) * (a.x - b.x) - (a.y - b.y) * (a.y - b.y) - (a.z - b.z) * (a.z - b.z);
    }

    CUDA_CALL CC_CONSTEXPR inline float distance(const vec3& a, const vec3& b)
    {
        return sqrtf(distance2(a, b));
    }

    CUDA_CALL constexpr inline vec3 cross(const vec3& a, const vec3& b)
    {
        return vec3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
    }

    CUDA_CALL CC_CONSTEXPR inline vec3 normalize(const vec3& a)
    {
        return a / length(a);
    }

    CUDA_CALL constexpr inline vec3 reflect(const vec3& I, const vec3& N)
    {
        return I - N * dot(N, I) * 2.f;
    }

    CUDA_CALL constexpr inline vec3 refract(const vec3& I, const vec3& N, float eta)
    {
        const float NdotI = dot(N, I);
        const float k = 1.f - eta * eta * (1.f - NdotI * NdotI);

        return (k >= .0f)? vec3(eta * I - (eta * NdotI + sqrtf(k)) * N) : vec3();
    }

    CUDA_CALL constexpr inline float determinant(const mat3& m)
    {
        return (+ m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
                - m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])
                + m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]));
    }

    CUDA_CALL constexpr inline mat3 inverse(const mat3& m)
    {
        const float one_over_det = 1.f / determinant(m);

        mat3 inverse;
        inverse[0][0] = + (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * one_over_det;
        inverse[1][0] = - (m[1][0] * m[2][2] - m[2][0] * m[1][2]) * one_over_det;
        inverse[2][0] = + (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * one_over_det;

        inverse[0][1] = - (m[0][1] * m[2][2] - m[2][1] * m[0][2]) * one_over_det;
        inverse[1][1] = + (m[0][0] * m[2][2] - m[2][0] * m[0][2]) * one_over_det;
        inverse[2][1] = - (m[0][0] * m[2][1] - m[2][0] * m[0][1]) * one_over_det;

        inverse[0][2] = + (m[0][1] * m[1][2] - m[1][1] * m[0][2]) * one_over_det;
        inverse[1][2] = - (m[0][0] * m[1][2] - m[1][0] * m[0][2]) * one_over_det;
        inverse[2][2] = + (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * one_over_det;

        return inverse;
    }

    CUDA_CALL constexpr inline mat3 transpose(const mat3& m)
    {
        mat3 transpose;
        transpose[0][0] = m[0][0];
        transpose[0][1] = m[1][0];
        transpose[0][2] = m[2][0];

        transpose[1][0] = m[0][1];
        transpose[1][1] = m[1][1];
        transpose[1][2] = m[2][1];

        transpose[2][0] = m[0][2];
        transpose[2][1] = m[1][2];
        transpose[2][2] = m[2][2];

        return transpose;
    }

    CUDA_CALL constexpr inline mat4 inverse(const mat4& m)
	{
		float coef00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
		float coef02 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
		float coef03 = m[1][2] * m[2][3] - m[2][2] * m[1][3];

		float coef04 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
		float coef06 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
		float coef07 = m[1][1] * m[2][3] - m[2][1] * m[1][3];

		float coef08 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
		float coef10 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
		float coef11 = m[1][1] * m[2][2] - m[2][1] * m[1][2];

		float coef12 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
		float coef14 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
		float coef15 = m[1][0] * m[2][3] - m[2][0] * m[1][3];

		float coef16 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
		float coef18 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
		float coef19 = m[1][0] * m[2][2] - m[2][0] * m[1][2];

		float coef20 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
		float coef22 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
		float coef23 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

		vec4 fac0(coef00, coef00, coef02, coef03);
		vec4 fac1(coef04, coef04, coef06, coef07);
		vec4 fac2(coef08, coef08, coef10, coef11);
		vec4 fac3(coef12, coef12, coef14, coef15);
		vec4 fac4(coef16, coef16, coef18, coef19);
		vec4 fac5(coef20, coef20, coef22, coef23);

		vec4 vec0(m[1][0], m[0][0], m[0][0], m[0][0]);
		vec4 vec1(m[1][1], m[0][1], m[0][1], m[0][1]);
		vec4 vec2(m[1][2], m[0][2], m[0][2], m[0][2]);
		vec4 vec3(m[1][3], m[0][3], m[0][3], m[0][3]);

		vec4 inv0(vec1 * fac0 - vec2 * fac1 + vec3 * fac2);
		vec4 inv1(vec0 * fac0 - vec2 * fac3 + vec3 * fac4);
		vec4 inv2(vec0 * fac1 - vec1 * fac3 + vec3 * fac5);
		vec4 inv3(vec0 * fac2 - vec1 * fac4 + vec2 * fac5);

		vec4 sign_a(+1, -1, +1, -1);
		vec4 sign_b(-1, +1, -1, +1);

		mat4 inv{ inv0 * sign_a, inv1 * sign_b, inv2 * sign_a, inv3 * sign_b };
		
		vec4 row0(inv[0][0], inv[1][0], inv[2][0], inv[3][0]);

		vec4 dot0(m[0] * row0);
		float dot1 = (dot0.x + dot0.y) + (dot0.z + dot0.w);
		float one_over_det = 1.f / dot1;

		return mat4{ inv[0] * one_over_det, inv[1] * one_over_det, inv[2] * one_over_det, inv[3] * one_over_det };
	}

    CUDA_CALL constexpr inline mat4 transpose(const mat4& m)
    {
        mat4 transpose;
        transpose[0][0] = m[0][0];
        transpose[0][1] = m[1][0];
        transpose[0][2] = m[2][0];
        transpose[0][3] = m[3][0];

        transpose[1][0] = m[0][1];
        transpose[1][1] = m[1][1];
        transpose[1][2] = m[2][1];
        transpose[1][3] = m[3][1];

        transpose[2][0] = m[0][2];
        transpose[2][1] = m[1][2];
        transpose[2][2] = m[2][2];
        transpose[2][3] = m[3][2];

        transpose[3][0] = m[0][3];
        transpose[3][1] = m[1][3];
        transpose[3][2] = m[2][3];
        transpose[3][3] = m[3][3];

        return transpose;
    }

    CUDA_CALL constexpr inline mat4 translate(const mat4& m, const vec3& v)
    {
        return mat4
        {
            m[0],
            m[1],
            m[2],
            m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3]
        };
    }

    CUDA_CALL CC_CONSTEXPR inline mat4 rotate(const mat4& m, float angle, const vec3& axis)
    {
        const vec3 axis_n = normalize(axis);
        const float x = axis_n.x;
        const float y = axis_n.y;
        const float z = axis_n.z;
        const float c = cosf(angle);
        const float s = sinf(angle);

        const mat4 rot
        {
            vec4{x * x * (1.f - c) + c,      y * x * (1.f - c) + z * s,  x * z * (1.f - c) - y * s,  0},
            vec4{x * y * (1.f - c) - z * s,  y * y * (1.f - c) + c,      y * z * (1.f - c) + x * s,  0},
            vec4{x * z * (1.f - c) + y * s,  y * z * (1.f - c) - x * s,  z * z * (1.f - c) + c,      0},
            vec4{0,                          0,                          0,                          1}
        };

        return m * rot;
    }

    CUDA_CALL constexpr inline mat4 scale(const mat4& m, const vec3& v)
    {
        return mat4
        {
            m[0] * v.x,
            m[1] * v.y,
            m[2] * v.z,
            m[3]
        };
    }

    CUDA_CALL CC_CONSTEXPR inline mat4 lookAt(const vec3& eye, const vec3& center, const vec3& up)
    {
        const vec3 f(normalize(center - eye));
        const vec3 s(normalize(cross(f, up)));
        const vec3 u(cross(s, f));

        return mat4
        {
            vec4{          s.x,          u.x,          -f.x,  0.0f },
            vec4{          s.y,          u.y,          -f.y,  0.0f },
            vec4{          s.z,          u.z,          -f.z,  0.0f },
            vec4{ -dot(s, eye), -dot(u, eye),   dot(f, eye),  1.0f }
        };
    }

    CUDA_CALL CC_CONSTEXPR inline mat4 perspective(float fovy, float aspect, float znear, float zfar)
    {
        const float f = rcp(tanf(fovy / 2.0f));
		return mat4
        {
            vec4{ f / aspect,  0.0f,                                    0.0f,   0.0f },
            vec4{       0.0f,     f,                                    0.0f,   0.0f },
            vec4{       0.0f,  0.0f,        -(zfar + znear) / (zfar - znear),  -1.0f },
            vec4{       0.0f,  0.0f,  -(2.f * zfar * znear) / (zfar - znear),   0.0f }
        };
    }
}

namespace gfx
{
using math::vec3;
using math::vec4;
using math::pow;

CUDA_CALL CC_CONSTEXPR inline float srgb(float linear)
{
    return (linear <= 0.0031308f)? 12.92f * linear : 1.055f * pow(linear, 1.f / 2.4f) - 0.055f;
}

CUDA_CALL CC_CONSTEXPR inline float linear(float srgb)
{
    return (srgb <= 0.04045f) ? srgb / 12.92f : pow((srgb + 0.055f) / 1.055f, 2.4f);
}

CUDA_CALL CC_CONSTEXPR inline vec3 srgb(const vec3& linear)   { return vec3(srgb(linear.r), srgb(linear.g), srgb(linear.b)); }
CUDA_CALL CC_CONSTEXPR inline vec4 srgb(const vec4& linear)   { return vec4(srgb(vec3(linear.rgb)), linear.a); }

CUDA_CALL CC_CONSTEXPR inline vec3 linear(const vec3& srgb)   { return vec3(linear(srgb.r), linear(srgb.g), linear(srgb.b)); }
CUDA_CALL CC_CONSTEXPR inline vec4 linear(const vec4& srgb)   { return vec4(linear(vec3(srgb.rgb)), srgb.a); }

CUDA_CALL CC_CONSTEXPR inline float aces(float x)
{
    // Narkowicz, "ACES Filmic Tone Mapping Curve"
    // https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    constexpr float a = 2.51f;
    constexpr float b = 0.03f;
    constexpr float c = 2.43f;
    constexpr float d = 0.59f;
    constexpr float e = 0.14f;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

CUDA_CALL CC_CONSTEXPR inline vec3 aces(const vec3& linear) { return vec3(aces(linear.x), aces(linear.y), aces(linear.z)); }
CUDA_CALL CC_CONSTEXPR inline vec4 aces(const vec4& linear) { return vec4(aces(linear.x), aces(linear.y), aces(linear.z), linear.a); }

CUDA_CALL CC_CONSTEXPR inline float reinhard(float x)
{
    return x / (1.0f + x);
}

CUDA_CALL CC_CONSTEXPR inline vec3 reinhard(const vec3& linear) { return vec3(reinhard(linear.x), reinhard(linear.y), reinhard(linear.z)); }
CUDA_CALL CC_CONSTEXPR inline vec4 reinhard(const vec4& linear) { return vec4(reinhard(linear.x), reinhard(linear.y), reinhard(linear.z), linear.a); }

}

namespace yuv
{
using math::vec3;

CUDA_CALL CC_CONSTEXPR inline vec3 yuv(vec3 rgb)
{
    vec3 yuv;

    yuv.x = .299f * rgb.r + .587f * rgb.g + .114f * rgb.b;
    yuv.y = (rgb.b - yuv.x) * .565f;
    yuv.z = (rgb.r - yuv.x) * .713f;

    return yuv;
}

CUDA_CALL CC_CONSTEXPR inline vec3 rgb(vec3 yuv)
{
    vec3 rgb;

    rgb.r = yuv.x + 1.403f * yuv.z;
    rgb.g = yuv.x -  .344f * yuv.y - .714f * yuv.z;
    rgb.b = yuv.x + 1.770f * yuv.y;

    return rgb;
}

}

}
