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
#include <limits>
#include <cassert>
#include <cmath>
#include <initializer_list>

#if defined(_MSC_VER)
 #include <intrin.h>
#else
 #include <x86intrin.h>
#endif

namespace cc
{
namespace util
{
    //
    // useful functions
    //
    template<typename T>
    constexpr inline const T& min(const T& a, const T& b) { return !(b < a)? a : b; }

    template<typename T>
    constexpr inline const T& max(const T& a, const T& b) { return (a < b)? b : a; }

    template<typename T>
    constexpr inline const T& clamp(const T& a, const T& lower, const T& upper) { return util::min(util::max(a, lower), upper); }

    template<typename T>
    constexpr inline const T& saturate(const T& a) { return clamp(a, T(0), T(1)); }

    template<typename T, size_t N>
    constexpr inline uint32_t array_size(const T(&)[N]) { return N; }

    template<typename T>
    constexpr inline void swap(T& a, T& b) { T tmp(a); a = b; b = tmp; }

    template<typename T>
    constexpr inline T sign(const T& x) { return T((x > T(0)) - (x < T(0))); }

    template<typename T>
    constexpr inline T abs(const T& a) { return (a < T(0)) ? -a : a; }
}

namespace math
{
    //
    // math constants
    //
    constexpr float PI = 3.1415926535897932f;
    constexpr float PI_2 = 1.57079632679f;
    constexpr float EPS = 1.e-8f;

namespace fast
{
    //
    // (hopefully) faster but less accurate implementations than stdlib
    //
    inline float rcp(float x)
    {
        return _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(x)));
    }
        
    inline float rsqrt(float x)
    {
        float rsqrt_tmp = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
        return rsqrt_tmp * (1.5f - x * 0.5f * rsqrt_tmp * rsqrt_tmp);
    }

    constexpr inline float atan2f(float y, float x)
    {
        constexpr float c1 = PI / 4.f;
        constexpr float c2 = PI * 3.f / 4.f;

        float result = .0f;
        if (y != 0 || x != 0)
        {
            float angle = (x >= .0f)? c1 - c1 * ((x - fabsf(y)) / (x + fabsf(y))) :
                                      c2 - c1 * ((x + fabsf(y)) / (fabsf(y) - x));
            
            result = (y < .0f)? -angle : angle;
        }

        return result;
    }

    constexpr float SINLUT[] = {
        +0.000000f, +0.012272f, +0.024541f, +0.036807f, +0.049068f, +0.061321f, +0.073565f, +0.085797f, +0.098017f, +0.110222f, +0.122411f, +0.134581f, +0.146730f, +0.158858f, +0.170962f, +0.183040f,
        +0.195090f, +0.207111f, +0.219101f, +0.231058f, +0.242980f, +0.254866f, +0.266713f, +0.278520f, +0.290285f, +0.302006f, +0.313682f, +0.325310f, +0.336890f, +0.348419f, +0.359895f, +0.371317f,
        +0.382683f, +0.393992f, +0.405241f, +0.416430f, +0.427555f, +0.438616f, +0.449611f, +0.460539f, +0.471397f, +0.482184f, +0.492898f, +0.503538f, +0.514103f, +0.524590f, +0.534998f, +0.545325f,
        +0.555570f, +0.565732f, +0.575808f, +0.585798f, +0.595699f, +0.605511f, +0.615232f, +0.624860f, +0.634393f, +0.643832f, +0.653173f, +0.662416f, +0.671559f, +0.680601f, +0.689541f, +0.698376f,
        +0.707107f, +0.715731f, +0.724247f, +0.732654f, +0.740951f, +0.749136f, +0.757209f, +0.765167f, +0.773010f, +0.780737f, +0.788346f, +0.795837f, +0.803208f, +0.810457f, +0.817585f, +0.824589f,
        +0.831470f, +0.838225f, +0.844854f, +0.851355f, +0.857729f, +0.863973f, +0.870087f, +0.876070f, +0.881921f, +0.887640f, +0.893224f, +0.898674f, +0.903989f, +0.909168f, +0.914210f, +0.919114f,
        +0.923880f, +0.928506f, +0.932993f, +0.937339f, +0.941544f, +0.945607f, +0.949528f, +0.953306f, +0.956940f, +0.960431f, +0.963776f, +0.966976f, +0.970031f, +0.972940f, +0.975702f, +0.978317f,
        +0.980785f, +0.983105f, +0.985278f, +0.987301f, +0.989177f, +0.990903f, +0.992480f, +0.993907f, +0.995185f, +0.996313f, +0.997290f, +0.998118f, +0.998795f, +0.999322f, +0.999699f, +0.999925f,
        +1.000000f, +0.999925f, +0.999699f, +0.999322f, +0.998795f, +0.998118f, +0.997290f, +0.996313f, +0.995185f, +0.993907f, +0.992480f, +0.990903f, +0.989177f, +0.987301f, +0.985278f, +0.983105f,
        +0.980785f, +0.978317f, +0.975702f, +0.972940f, +0.970031f, +0.966976f, +0.963776f, +0.960431f, +0.956940f, +0.953306f, +0.949528f, +0.945607f, +0.941544f, +0.937339f, +0.932993f, +0.928506f,
        +0.923880f, +0.919114f, +0.914210f, +0.909168f, +0.903989f, +0.898674f, +0.893224f, +0.887640f, +0.881921f, +0.876070f, +0.870087f, +0.863973f, +0.857729f, +0.851355f, +0.844854f, +0.838225f,
        +0.831470f, +0.824589f, +0.817585f, +0.810457f, +0.803207f, +0.795837f, +0.788346f, +0.780737f, +0.773010f, +0.765167f, +0.757209f, +0.749136f, +0.740951f, +0.732654f, +0.724247f, +0.715731f,
        +0.707107f, +0.698376f, +0.689540f, +0.680601f, +0.671559f, +0.662416f, +0.653173f, +0.643831f, +0.634393f, +0.624859f, +0.615232f, +0.605511f, +0.595699f, +0.585798f, +0.575808f, +0.565732f,
        +0.555570f, +0.545325f, +0.534998f, +0.524590f, +0.514103f, +0.503538f, +0.492898f, +0.482184f, +0.471397f, +0.460539f, +0.449611f, +0.438616f, +0.427555f, +0.416429f, +0.405241f, +0.393992f,
        +0.382683f, +0.371317f, +0.359895f, +0.348419f, +0.336890f, +0.325310f, +0.313682f, +0.302006f, +0.290285f, +0.278520f, +0.266713f, +0.254866f, +0.242980f, +0.231058f, +0.219101f, +0.207111f,
        +0.195090f, +0.183040f, +0.170962f, +0.158858f, +0.146730f, +0.134581f, +0.122411f, +0.110222f, +0.098017f, +0.085797f, +0.073564f, +0.061321f, +0.049068f, +0.036807f, +0.024541f, +0.012271f,
        -0.000000f, -0.012272f, -0.024541f, -0.036807f, -0.049068f, -0.061321f, -0.073565f, -0.085797f, -0.098017f, -0.110222f, -0.122411f, -0.134581f, -0.146731f, -0.158858f, -0.170962f, -0.183040f,
        -0.195090f, -0.207111f, -0.219101f, -0.231058f, -0.242980f, -0.254866f, -0.266713f, -0.278520f, -0.290285f, -0.302006f, -0.313682f, -0.325310f, -0.336890f, -0.348419f, -0.359895f, -0.371317f,
        -0.382684f, -0.393992f, -0.405241f, -0.416430f, -0.427555f, -0.438616f, -0.449611f, -0.460539f, -0.471397f, -0.482184f, -0.492898f, -0.503538f, -0.514103f, -0.524590f, -0.534998f, -0.545325f,
        -0.555570f, -0.565732f, -0.575808f, -0.585798f, -0.595699f, -0.605511f, -0.615232f, -0.624860f, -0.634393f, -0.643832f, -0.653173f, -0.662416f, -0.671559f, -0.680601f, -0.689541f, -0.698376f,
        -0.707107f, -0.715731f, -0.724247f, -0.732654f, -0.740951f, -0.749136f, -0.757209f, -0.765167f, -0.773011f, -0.780737f, -0.788346f, -0.795837f, -0.803208f, -0.810457f, -0.817585f, -0.824589f,
        -0.831470f, -0.838225f, -0.844854f, -0.851355f, -0.857729f, -0.863973f, -0.870087f, -0.876070f, -0.881921f, -0.887640f, -0.893224f, -0.898675f, -0.903989f, -0.909168f, -0.914210f, -0.919114f,
        -0.923880f, -0.928506f, -0.932993f, -0.937339f, -0.941544f, -0.945607f, -0.949528f, -0.953306f, -0.956940f, -0.960431f, -0.963776f, -0.966977f, -0.970031f, -0.972940f, -0.975702f, -0.978317f,
        -0.980785f, -0.983106f, -0.985278f, -0.987301f, -0.989177f, -0.990903f, -0.992480f, -0.993907f, -0.995185f, -0.996313f, -0.997290f, -0.998118f, -0.998795f, -0.999322f, -0.999699f, -0.999925f,
        -1.000000f, -0.999925f, -0.999699f, -0.999322f, -0.998795f, -0.998118f, -0.997290f, -0.996313f, -0.995185f, -0.993907f, -0.992480f, -0.990903f, -0.989176f, -0.987301f, -0.985278f, -0.983105f,
        -0.980785f, -0.978317f, -0.975702f, -0.972940f, -0.970031f, -0.966976f, -0.963776f, -0.960430f, -0.956940f, -0.953306f, -0.949528f, -0.945607f, -0.941544f, -0.937339f, -0.932993f, -0.928506f,
        -0.923879f, -0.919114f, -0.914210f, -0.909168f, -0.903989f, -0.898674f, -0.893224f, -0.887640f, -0.881921f, -0.876070f, -0.870087f, -0.863973f, -0.857729f, -0.851355f, -0.844853f, -0.838225f,
        -0.831470f, -0.824589f, -0.817585f, -0.810457f, -0.803207f, -0.795837f, -0.788346f, -0.780737f, -0.773010f, -0.765167f, -0.757209f, -0.749136f, -0.740951f, -0.732654f, -0.724247f, -0.715731f,
        -0.707107f, -0.698376f, -0.689540f, -0.680601f, -0.671559f, -0.662416f, -0.653173f, -0.643831f, -0.634393f, -0.624859f, -0.615231f, -0.605511f, -0.595699f, -0.585798f, -0.575808f, -0.565732f,
        -0.555570f, -0.545325f, -0.534997f, -0.524590f, -0.514103f, -0.503538f, -0.492898f, -0.482184f, -0.471397f, -0.460539f, -0.449611f, -0.438616f, -0.427555f, -0.416429f, -0.405241f, -0.393992f,
        -0.382683f, -0.371317f, -0.359895f, -0.348419f, -0.336890f, -0.325310f, -0.313682f, -0.302006f, -0.290285f, -0.278520f, -0.266713f, -0.254865f, -0.242980f, -0.231058f, -0.219101f, -0.207111f,
        -0.195090f, -0.183040f, -0.170962f, -0.158858f, -0.146730f, -0.134581f, -0.122411f, -0.110222f, -0.098017f, -0.085797f, -0.073564f, -0.061321f, -0.049068f, -0.036807f, -0.024541f, -0.012271f
    };
    constexpr int MAX_ANGLE_SAMPLES = util::array_size(SINLUT);

    constexpr inline float sinf(float x)
    {
        float f = x * (MAX_ANGLE_SAMPLES / 2) / PI;
        if (f < 0.f)
        {
            return SINLUT[(-((-int(f)) & (MAX_ANGLE_SAMPLES - 1))) + MAX_ANGLE_SAMPLES];
            
        }
        else
        {
            return SINLUT[int(f) & (MAX_ANGLE_SAMPLES - 1)];
        }
    }

    constexpr inline float cosf(float x)
    {
        float f = x * (MAX_ANGLE_SAMPLES / 2) / PI;
        if (f < 0.f)
        {
            return SINLUT[((-int(f)) + (MAX_ANGLE_SAMPLES / 4)) & (MAX_ANGLE_SAMPLES - 1)];
        }
        else
        {
            return SINLUT[(int(f) + (MAX_ANGLE_SAMPLES / 4)) & (MAX_ANGLE_SAMPLES - 1)];
        }
    }
}


    //
    // conversion utils
    //
    constexpr inline float radians(float deg)                { return deg * PI / 180.0f; }
    constexpr inline float degrees(float rad)                { return rad * 180.0f / PI; }
    
    template<typename T>
    constexpr inline T lerp(const T& v0, const T& v1, float t) { return v0 * t + v1 * (1.f - t); }


    // cotangent
    constexpr inline float cot(float x) { return fast::cosf(x) / fast::sinf(x); }


    //
    // useful types
    //
    struct vec2
    {
        union {
            float v[2];
            struct { float x, y; };
            struct { float s, t; };
            struct { float w, h; };
        };

        constexpr inline float& operator[](size_t i)             { assert(i < 2); return v[i]; }
        constexpr inline const float& operator[](size_t i) const { assert(i < 2); return v[i]; }

        constexpr inline vec2() : v{} {}
        constexpr inline explicit vec2(float _v) : v{_v, _v} {}
        constexpr inline explicit vec2(float _v1, float _v2) : v{ _v1, _v2 } {}
        constexpr inline explicit vec2(std::initializer_list<float> _v) : v{*_v.begin(), *(_v.begin() + 1)} { assert(_v.size() == 2); }
    };

    struct vec3
    {
        union {
            float v[3];
            struct { float x, y, z; };
            struct { float r, g, b; };
        };

        constexpr inline float& operator[](size_t i)             { assert(i < 3); return v[i]; }
        constexpr inline const float& operator[](size_t i) const { assert(i < 3); return v[i]; }

        constexpr inline vec3() : v{} {}
        constexpr inline explicit vec3(float _v) : v{_v, _v, _v} {}
        constexpr inline explicit vec3(float _v1, float _v2, float _v3) : v{ _v1, _v2, _v3 } {}
        constexpr inline explicit vec3(std::initializer_list<float> _v) : v{*_v.begin(), *(_v.begin() + 1), *(_v.begin() + 2)} { assert(_v.size() == 3); }
    };

    struct vec4
    {
        union {
            float v[4];
            struct { float x, y, z, w; };
            struct { float r, g, b, a; };
        };

        constexpr inline float& operator[](size_t i)             { assert(i < 4); return v[i]; }
        constexpr inline const float& operator[](size_t i) const { assert(i < 4); return v[i]; }

        constexpr inline vec4() : v{} {}
        constexpr inline explicit vec4(float _v) : v{_v, _v, _v, _v} {}
        constexpr inline explicit vec4(float _v1, float _v2, float _v3, float _v4) : v{ _v1, _v2, _v3, _v4 } {}
        constexpr inline explicit vec4(std::initializer_list<float> _v) : v{*_v.begin(), *(_v.begin() + 1), *(_v.begin() + 2), *(_v.begin() + 3)} { assert(_v.size() == 4); }
    };

    struct mat4
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

        constexpr vec4& operator[](size_t i)             { assert(i < 4); return m[i]; }
        constexpr const vec4& operator[](size_t i) const { assert(i < 4); return m[i]; }

        constexpr inline mat4() : m{} {}
        constexpr inline explicit mat4(float _i) : m{} { _m00 = _m11 =_m22 = _m33 = _i; }
        constexpr inline explicit mat4(std::initializer_list<vec4> _m) : m{*_m.begin(), *(_m.begin() + 1), *(_m.begin() + 2), *(_m.begin() + 3)} { assert(_m.size() == 4); }
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

        constexpr vec3& operator[](size_t i)             { assert(i < 3); return m[i]; }
        constexpr const vec3& operator[](size_t i) const { assert(i < 3); return m[i]; }

        constexpr inline mat3() : m{} {}
        constexpr inline explicit mat3(float _i) : m{} { _m00 = _m11 =_m22 = _i; }
        constexpr inline explicit mat3(const mat4& _m) : _m00(_m._m00), _m10(_m._m10), _m20(_m._m20), _m01(_m._m01), _m11(_m._m11), _m21(_m._m21), _m02(_m._m02), _m12(_m._m12), _m22(_m._m22) {}
        constexpr inline explicit mat3(std::initializer_list<vec3> _m) : m{*_m.begin(), *(_m.begin() + 1), *(_m.begin() + 2)} { assert(_m.size() == 3); }
    };

    //
    // compatibility with GLM
    //
    constexpr inline const float* value_ptr(const vec2& v)                   { return &(v.v[0]); }
    constexpr inline const float* value_ptr(const vec3& v)                   { return &(v.v[0]); }
    constexpr inline const float* value_ptr(const vec4& v)                   { return &(v.v[0]); }
    constexpr inline const float* value_ptr(const mat3& m)                   { return value_ptr(m.m[0]); }
    constexpr inline const float* value_ptr(const mat4& m)                   { return value_ptr(m.m[0]); }

    //
    // operators
    //
    constexpr inline vec2 operator+(const vec2& a, float b)                  { return vec2{ a.x + b, a.y + b }; }
    constexpr inline vec2 operator+(float b, const vec2& a)                  { return vec2{ a.x + b, a.y + b }; }
    constexpr inline vec2 operator+(const vec2& a, const vec2& b)            { return vec2{ a.x + b.x, a.y + b.y }; }
    constexpr inline vec2 operator-(const vec2& a)                           { return vec2{ -a.x, -a.y }; }
    constexpr inline vec2 operator-(const vec2& a, float b)                  { return vec2{ a.x - b, a.y - b }; }
    constexpr inline vec2 operator-(float b, const vec2& a)                  { return vec2{ a.x - b, a.y - b }; }
    constexpr inline vec2 operator-(const vec2& a, const vec2& b)            { return vec2{ a.x - b.x, a.y - b.y }; }
    constexpr inline vec2 operator*(const vec2& a, float b)                  { return vec2{ a.x * b, a.y * b }; }
    constexpr inline vec2 operator*(float b, const vec2& a)                  { return vec2{ a.x * b, a.y * b }; }
    constexpr inline vec2 operator*(const vec2& a, const vec2& b)            { return vec2{ a.x * b.x, a.y * b.y }; }
    constexpr inline vec2 operator/(const vec2& a, float b)                  { return vec2{ a.x / b, a.y / b }; }
    constexpr inline vec2 operator/(float a, const vec2& b)                  { return vec2{ a / b.x, a / b.y }; }
    constexpr inline vec2 operator/(const vec2& a, const vec2& b)            { return vec2{ a.x / b.x, a.y / b.y }; }
    constexpr inline vec2& operator+=(vec2& a, float b)                      { a.x += b; a.y += b; return a; }
    constexpr inline vec2& operator-=(vec2& a, float b)                      { a.x -= b; a.y -= b; return a; }
    constexpr inline vec2& operator*=(vec2& a, float b)                      { a.x *= b; a.y *= b; return a; }
    constexpr inline vec2& operator/=(vec2& a, float b)                      { a.x /= b; a.y /= b; return a; }
    constexpr inline bool operator==(const vec2& a, const vec2& b)           { return util::abs(a.x - b.x) < EPS && util::abs(a.y - b.y) < EPS; }
    constexpr inline bool operator!=(const vec2& a, const vec2& b)           { return !(a == b); }

    constexpr inline vec3 operator+(const vec3& a, float b)                  { return vec3{ a.x + b, a.y + b, a.z + b }; }
    constexpr inline vec3 operator+(float b, const vec3& a)                  { return vec3{ a.x + b, a.y + b, a.z + b }; }
    constexpr inline vec3 operator+(const vec3& a, const vec3& b)            { return vec3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
    constexpr inline vec3 operator-(const vec3& a)                           { return vec3{ -a.x, -a.y, -a.z }; }
    constexpr inline vec3 operator-(const vec3& a, float b)                  { return vec3{ a.x - b, a.y - b, a.z - b }; }
    constexpr inline vec3 operator-(float b, const vec3& a)                  { return vec3{ a.x - b, a.y - b, a.z - b }; }
    constexpr inline vec3 operator-(const vec3& a, const vec3& b)            { return vec3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
    constexpr inline vec3 operator*(const vec3& a, float b)                  { return vec3{ a.x * b, a.y * b, a.z * b }; }
    constexpr inline vec3 operator*(float b, const vec3& a)                  { return vec3{ a.x * b, a.y * b, a.z * b }; }
    constexpr inline vec3 operator*(const vec3& a, const vec3& b)            { return vec3{ a.x * b.x, a.y * b.y, a.z * b.z }; }
    constexpr inline vec3 operator/(const vec3& a, float b)                  { return vec3{ a.x / b, a.y / b, a.z / b }; }
    constexpr inline vec3 operator/(float a, const vec3& b)                  { return vec3{ a / b.x, a / b.y, a / b.z }; }
    constexpr inline vec3 operator/(const vec3& a, const vec3& b)            { return vec3{ a.x / b.x, a.y / b.y, a.z / b.z }; }
    constexpr inline vec3& operator+=(vec3& a, const vec3& b)                { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
    constexpr inline vec3& operator-=(vec3& a, const vec3& b)                { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
    constexpr inline vec3& operator*=(vec3& a, const vec3& b)                { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
    constexpr inline vec3& operator/=(vec3& a, const vec3& b)                { a.x /= b.x; a.y /= b.y; a.z /= b.z; return a; }
    constexpr inline vec3& operator+=(vec3& a, float b)                      { a.x += b; a.y += b; a.z += b; return a; }
    constexpr inline vec3& operator-=(vec3& a, float b)                      { a.x -= b; a.y -= b; a.z -= b; return a; }
    constexpr inline vec3& operator*=(vec3& a, float b)                      { a.x *= b; a.y *= b; a.z *= b; return a; }
    constexpr inline vec3& operator/=(vec3& a, float b)                      { a.x /= b; a.y /= b; a.z /= b; return a; }
    constexpr inline bool operator==(const vec3& a, const vec3& b)           { return util::abs(a.x - b.x) < EPS && util::abs(a.y - b.y) < EPS && util::abs(a.z - b.z) < EPS; }
    constexpr inline bool operator!=(const vec3& a, const vec3& b)           { return !(a == b); }

    constexpr inline vec4 operator+(const vec4& a, float b)                  { return vec4{ a.x + b, a.y + b, a.z + b, a.w + b }; }
    constexpr inline vec4 operator+(float b, const vec4& a)                  { return vec4{ a.x + b, a.y + b, a.z + b, a.w + b }; }
    constexpr inline vec4 operator+(const vec4& a, const vec4& b)            { return vec4{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }
    constexpr inline vec4 operator-(const vec4& a)                           { return vec4{ -a.x, -a.y, -a.z, -a.w }; }
    constexpr inline vec4 operator-(const vec4& a, float b)                  { return vec4{ a.x - b, a.y - b, a.z - b, a.w - b }; }
    constexpr inline vec4 operator-(float b, const vec4& a)                  { return vec4{ a.x - b, a.y - b, a.z - b, a.w - b }; }
    constexpr inline vec4 operator-(const vec4& a, const vec4& b)            { return vec4{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }
    constexpr inline vec4 operator*(const vec4& a, float b)                  { return vec4{ a.x * b, a.y * b, a.z * b, a.w * b }; }
    constexpr inline vec4 operator*(float b, const vec4& a)                  { return vec4{ a.x * b, a.y * b, a.z * b, a.w * b }; }
    constexpr inline vec4 operator*(const vec4& a, const vec4& b)            { return vec4{ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w }; }
    constexpr inline vec4 operator/(const vec4& a, float b)                  { return vec4{ a.x / b, a.y / b, a.z / b, a.w / b }; }
    constexpr inline vec4 operator/(float a, const vec4& b)                  { return vec4{ a / b.x, a / b.y, a / b.z, a / b.w }; }
    constexpr inline vec4 operator/(const vec4& a, const vec4& b)            { return vec4{ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w }; }
    constexpr inline vec4& operator+=(vec4& a, const vec4& b)                { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
    constexpr inline vec4& operator-=(vec4& a, const vec4& b)                { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
    constexpr inline vec4& operator*=(vec4& a, const vec4& b)                { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }
    constexpr inline vec4& operator/=(vec4& a, const vec4& b)                { a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w; return a; }
    constexpr inline vec4& operator+=(vec4& a, float b)                      { a.x += b; a.y += b; a.z += b; a.w += b; return a; }
    constexpr inline vec4& operator-=(vec4& a, float b)                      { a.x -= b; a.y -= b; a.z -= b; a.w -= b; return a; }
    constexpr inline vec4& operator*=(vec4& a, float b)                      { a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a; }
    constexpr inline vec4& operator/=(vec4& a, float b)                      { a.x /= b; a.y /= b; a.z /= b; a.w /= b; return a; }
    constexpr inline bool operator==(const vec4& a, const vec4& b)           { return util::abs(a.x - b.x) < EPS && util::abs(a.y - b.y) < EPS && util::abs(a.z - b.z) < EPS && util::abs(a.w - b.w) < EPS; }
    constexpr inline bool operator!=(const vec4& a, const vec4& b)           { return !(a == b); }

    constexpr inline mat3 operator*(const mat3& a, const mat3& b)
    {
        return mat3
        {
            { a[0] * b[0].x + a[1] * b[0].y + a[2] * b[0].z },
            { a[0] * b[1].x + a[1] * b[1].y + a[2] * b[1].z },
            { a[0] * b[2].x + a[1] * b[2].y + a[2] * b[2].z }
        };
    }

    constexpr inline vec3 operator*(const mat3& a, const vec3& b)
    {
        return vec3
        {
            a[0].x * b.x + a[0].y * b.y + a[0].z * b.z,
            a[1].x * b.x + a[1].y * b.y + a[1].z * b.z,
            a[2].x * b.x + a[2].y * b.y + a[2].z * b.z
        };
    }

    constexpr inline vec3 operator*(const vec3& a, const mat3& b)
    {
        return vec3
        {
            a.x * b[0].x + a.y * b[1].x + a.z * b[2].x,
            a.x * b[0].y + a.y * b[1].y + a.z * b[2].y,
            a.x * b[0].z + a.y * b[1].z + a.z * b[2].z
        };
    }

    constexpr inline mat4 operator*(const mat4& a, const mat4& b)
    {
        return mat4
        {
            { a[0] * b[0].x + a[1] * b[0].y + a[2] * b[0].z + a[3] * b[0].w },
            { a[0] * b[1].x + a[1] * b[1].y + a[2] * b[1].z + a[3] * b[1].w },
            { a[0] * b[2].x + a[1] * b[2].y + a[2] * b[2].z + a[3] * b[2].w },
            { a[0] * b[3].x + a[1] * b[3].y + a[2] * b[3].z + a[3] * b[3].w }
        };
    }


    //
    // trig functions
    //
    constexpr inline float dot(const vec3& a, const vec3& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    constexpr inline float length2(const vec3& a)
    {
        return dot(a, a);
    }

    inline float length(const vec3& a)
    {
        return sqrtf(length2(a));
    }

    constexpr inline float distance2(const vec3& a, const vec3& b)
    {
        return (a.x - b.x) * (a.x - b.x) - (a.y - b.y) * (a.y - b.y) - (a.z - b.z) * (a.z - b.z);
    }

    inline float distance(const vec3& a, const vec3& b)
    {
        return sqrtf(distance2(a, b));
    }

    constexpr inline vec3 cross(const vec3& a, const vec3& b)
    {
        return vec3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
    }

    inline vec3 normalize(const vec3& a)
    {
        return a / length(a);
    }

    constexpr inline vec3 reflect(const vec3& I, const vec3& N)
    {
        return I - N * dot(N, I) * 2.f;
    }

    constexpr inline vec3 refract(const vec3& I, const vec3& N, float eta)
    {
        const float NdotI = dot(N, I);
        const float k = 1.f - eta * eta * (1.f - NdotI * NdotI);

        return (k >= .0f)? vec3(eta * I - (eta * NdotI + sqrtf(k)) * N) : vec3();
    }

    constexpr inline mat4 translate(const mat4& m, const vec3& v)
    {
        return mat4
        {
            m[0],
            m[1],
            m[2],
            m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3]
        };
    }

    inline mat4 rotate(const mat4& m, float angle, const vec3& axis)
    {
        const vec3 axis_n = normalize(axis);
        const float x = axis_n.x;
        const float y = axis_n.y;
        const float z = axis_n.z;
        const float c = fast::cosf(angle);
        const float s = fast::sinf(angle);

        const mat4 rot
        {
            vec4{x * x * (1.f - c) + c,      y * x * (1.f - c) + z * s,  x * z * (1.f - c) - y * s,  0},
            vec4{x * y * (1.f - c) - z * s,  y * y * (1.f - c) + c,      y * z * (1.f - c) + x * s,  0},
            vec4{x * z * (1.f - c) + y * s,  y * z * (1.f - c) - x * s,  z * z * (1.f - c) + c,      0},
            vec4{0,                          0,                          0,                          1}
        };

        return m * rot;
    }

    constexpr inline mat4 scale(const mat4& m, const vec3& v)
    {
        return mat4
        {
            m[0] * v.x,
            m[1] * v.y,
            m[2] * v.z,
            m[3]
        };
    }

    inline mat4 lookAt(const vec3& eye, const vec3& center, const vec3& up)
    {
        const vec3 f(normalize(center - eye));
        const vec3 s(normalize(cross(up, f)));
        const vec3 u(cross(f, s));

        return mat4
        {
            vec4{          s.x,          u.x,          f.x,  0.0f },
            vec4{          s.y,          u.y,          f.y,  0.0f },
            vec4{          s.z,          u.z,          f.z,  0.0f },
            vec4{ -dot(s, eye), -dot(u, eye), -dot(f, eye),  1.0f }
        };
    }

    inline mat4 perspective(float fovy, float aspect, float znear, float zfar)
    {
        const float F = cot(fovy / 2.0f);
        const float delta = zfar - znear;

        return mat4
        {
            vec4{ F / aspect,   0.0f,                           0.0f,  0.0f },
            vec4{       0.0f,      F,                           0.0f,  0.0f },
            vec4{       0.0f,   0.0f,         (zfar + znear) / delta,  1.0f },
            vec4{       0.0f,   0.0f, -(2.0f * zfar * znear) / delta,  0.0f }
        };
    }
}
}
