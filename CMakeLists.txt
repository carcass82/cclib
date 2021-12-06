cmake_minimum_required(VERSION 3.8)

project(CCLib LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin/")

find_package(benchmark REQUIRED)
find_package(GTest REQUIRED)

find_package(glm REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(.)

set(BENCHMARK_SRC "./benchmark/benchmark.cpp")

set(TEST_SRC "./test/test.cpp")

#
# Compiler options
#
if (MSVC)
    message(STATUS "Found MSVC: " ${CMAKE_CXX_COMPILER})
    set(CMAKE_CXX_FLAGS "/nologo /W3 /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /EHsc /Zi /Od /D_DEBUG /DNOVTABLE= ")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi /GR- /GL /Gs- /O2 /fp:fast /DNDEBUG /DNOVTABLE=__declspec(novtable)")
	set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /LTCG /INCREMENTAL:NO /OPT:REF")
else()
    message(STATUS "Found C++ Compiler: " ${CMAKE_CXX_COMPILER})
    set(CMAKE_CXX_FLAGS "-Wall -no-pie -DNOVTABLE= ")
    set(CMAKE_CXX_FLAGS_DEBUG "-D_DEBUG -O0 -g")
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -Ofast -march=native -fsingle-precision-constant -fopenmp -fno-rtti -funroll-loops -D_GLIBCXX_PARALLEL")
	else()
		set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -Ofast -march=native -fsingle-precision-constant -fopenmp -flto -fno-rtti -funroll-loops -D_GLIBCXX_PARALLEL")
	endif()
endif()

message(STATUS "Debug CMAKE_CXX_FLAGS:  ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "Release CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "Debug CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
message(STATUS "Release CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_RELEASE}")

add_executable(${PROJECT_NAME}_Benchmark ${BENCHMARK_SRC})
target_link_libraries(${PROJECT_NAME}_Benchmark benchmark::benchmark)

add_executable(${PROJECT_NAME}_Test ${TEST_SRC})
target_link_libraries(${PROJECT_NAME}_Test GTest::GTest)
