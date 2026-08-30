cmake_minimum_required(VERSION 3.16)

project(nls_solve C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(FetchContent)

set(CMINPACK_PRECISION d CACHE STRING "" FORCE)
set(USE_BLAS OFF CACHE BOOL "" FORCE)
set(USE_LAPACK OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    cminpack
    GIT_REPOSITORY https://github.com/devernay/cminpack.git
    GIT_TAG v1.3.14
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cminpack)

add_library(nls_solve STATIC
    src/nls_solver.c
    src/gn_solver.c
    src/auglag_constrain.c
)

target_include_directories(nls_solve
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(nls_solve PUBLIC cminpack)

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(nls_solve PRIVATE -Wall -Wextra)
endif()

add_executable(test_nls tests/test_nls.c)
target_link_libraries(test_nls PRIVATE nls_solve)

add_executable(test_auglag tests/test_auglag.c)
target_link_libraries(test_auglag PRIVATE nls_solve)

add_executable(test_double_gaussian tests/test_double_gaussian.c)
target_link_libraries(test_double_gaussian PRIVATE nls_solve)

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(test_nls PRIVATE -Wall -Wextra)
    target_compile_options(test_auglag PRIVATE -Wall -Wextra)
    target_compile_options(test_double_gaussian PRIVATE -Wall -Wextra)
endif()

enable_testing()
add_test(NAME nls_core COMMAND test_nls)
add_test(NAME auglag_core_api COMMAND test_auglag)
add_test(NAME double_gaussian_golden COMMAND test_double_gaussian)
