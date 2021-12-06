#include <gtest/gtest.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "cclib.h"

// adjust tolerance for test results
static constexpr float EPS = 1.e-4f;

class Test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        float eye[] = { 2.f, 5.f, 10.f };
        float center[] = { .0f, .0f, .0f };
        float up[] = { .0f, 1.f, .0f };
        float fovy = 1.05f;
        float aspect = 1.33f;
        float near = .1f;
        float far = 1000.f;
        float vpos[] = { 3.f, 2.f, 1.f, 1.f };

        glm_V = glm::lookAt(glm::vec3(eye[0], eye[1], eye[2]),
                            glm::vec3(center[0], center[1], center[2]),
                            glm::vec3(up[0], up[1], up[2]));

        glm_P = glm::perspective(fovy, aspect, near, far);
        
        glm_P_V = glm_P * glm_V;
        
        glm_Pos = glm_P * glm_V * glm::vec4(vpos[0], vpos[1], vpos[2], vpos[3]);

        cc_V = cc::math::lookAt(cc::math::vec3(eye[0], eye[1], eye[2]),
                                cc::math::vec3(center[0], center[1], center[2]),
                                cc::math::vec3(up[0], up[1], up[2]));

        cc_P = cc::math::perspective(fovy, aspect, near, far);
        
        cc_P_V = cc_P * cc_V;
        
        cc_Pos = cc_P_V * cc::math::vec4(vpos[0], vpos[1], vpos[2], vpos[3]);
    }

    glm::mat4 glm_V, glm_P, glm_P_V;
    glm::vec4 glm_Pos;
    cc::math::mat4 cc_V, cc_P, cc_P_V;
    cc::math::vec4 cc_Pos;
};

TEST_F(Test, TestLookAt)
{
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            EXPECT_NEAR(glm_V[i][j], cc_V[i][j], EPS);
}

TEST_F(Test, TestPerspective)
{
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            EXPECT_NEAR(glm_P[i][j], cc_P[i][j], EPS);
}

TEST_F(Test, Test4x4MatrixMultiplication)
{
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            EXPECT_NEAR(glm_P_V[i][j], cc_P_V[i][j], EPS);
}

TEST_F(Test, TestMatrix4x4ByVec4)
{
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(glm_Pos[i], cc_Pos[i], EPS);
}

TEST_F(Test, Test3x3MatrixInverse)
{
    auto m_glm = glm::inverse(glm::mat3(glm_P_V));
    auto m_cc = cc::math::inverse(cc::math::mat3(cc_P_V));

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(m_glm[i][j], m_cc[i][j], EPS);
}

TEST_F(Test, Test4x4MatrixInverse)
{
    auto m_glm = glm::inverse(glm_P_V);
    auto m_cc = cc::math::inverse(cc_P_V);

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            EXPECT_NEAR(m_glm[i][j], m_cc[i][j], EPS);
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
