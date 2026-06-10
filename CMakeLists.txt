cmake_minimum_required(VERSION 3.12)
project(nested_cages)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-march=native)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/arch:AVX2 /fp:fast)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}$<0:>)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(Easy3D_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(Easy3D_BUILD_TUTORIALS OFF CACHE BOOL "" FORCE)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/Easy3D)
set(Easy3D_RESOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/Easy3D/resources)
set(LIBIGL_WITH_PREDICATES ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/ipc-toolkit)
add_subdirectory(third_party/Easy3D)
add_subdirectory(third_party/libigl)
add_subdirectory(third_party/cgal)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/ipc-toolkit/src)
add_executable(nested_cages ./src/main.cpp ./src/nested_cages.cpp)
target_include_directories(nested_cages PRIVATE 
    ${CMAKE_BINARY_DIR}/third_party/ipc-toolkit/include
)
target_compile_definitions(nested_cages PRIVATE 
    Easy3D_RESOURCE_DIR="${Easy3D_RESOURCE_DIR}"
	EASY3D_STATIC_DEFINE
)
set(Easy3D_DIR "${CMAKE_BINARY_DIR}/third_party/Easy3D")
find_package(OpenMP REQUIRED)
target_link_libraries(nested_cages
	easy3d::viewer
    easy3d::gui 
    easy3d::renderer 
    easy3d::algo 
    easy3d::core
	easy3d::util
	3rd_imgui
	Eigen3::Eigen
	igl::core
    ipc::toolkit
	spdlog::spdlog
	OpenMP::OpenMP_CXX
)

target_link_options(nested_cages PRIVATE "-Wl,--allow-multiple-definition")