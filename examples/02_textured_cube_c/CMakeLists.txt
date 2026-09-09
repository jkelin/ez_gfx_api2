cmake_minimum_required(VERSION 3.24)
project(ez_gfx_textured_cube LANGUAGES C)

include(FetchContent)

set(EZ_GFX_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH "ez_gfx_api2 repository root")
set(EZ_GFX_INCLUDE_DIR "${EZ_GFX_ROOT}/bindings/c/include" CACHE PATH "directory containing ez_gfx_api.h")
set(EZ_GFX_COMPILER "${EZ_GFX_ROOT}/target/debug/ez-gfx-compile${CMAKE_EXECUTABLE_SUFFIX}" CACHE FILEPATH "ez-gfx shader compiler")
if(WIN32)
    set(_EZ_GFX_FFI_LIBRARY "${EZ_GFX_ROOT}/target/debug/ez_gfx_ffi.dll.lib")
    set(_EZ_GFX_FFI_RUNTIME "${EZ_GFX_ROOT}/target/debug/ez_gfx_ffi.dll")
elseif(APPLE)
    set(_EZ_GFX_FFI_LIBRARY "${EZ_GFX_ROOT}/target/debug/libez_gfx_ffi.dylib")
    set(_EZ_GFX_FFI_RUNTIME "${_EZ_GFX_FFI_LIBRARY}")
else()
    set(_EZ_GFX_FFI_LIBRARY "${EZ_GFX_ROOT}/target/debug/libez_gfx_ffi.so")
    set(_EZ_GFX_FFI_RUNTIME "${_EZ_GFX_FFI_LIBRARY}")
endif()
set(EZ_GFX_FFI_LIBRARY "${_EZ_GFX_FFI_LIBRARY}" CACHE FILEPATH "ez-gfx FFI link library")
set(EZ_GFX_FFI_RUNTIME "${_EZ_GFX_FFI_RUNTIME}" CACHE FILEPATH "ez-gfx FFI runtime library")
set(EZ_GFX_SHADER_ARTIFACT "${EZ_GFX_ROOT}/target/c-examples/textured_cube/textured_cube.ezgfxshader" CACHE FILEPATH "generated shader artifact")
set(EZ_GFX_SHARED_SHADER_MODULE "${EZ_GFX_ROOT}/ez_gfx_api.slang" CACHE FILEPATH "shared ez-gfx shader module")

foreach(required EZ_GFX_INCLUDE_DIR EZ_GFX_COMPILER EZ_GFX_FFI_LIBRARY EZ_GFX_FFI_RUNTIME EZ_GFX_SHARED_SHADER_MODULE)
    if(NOT EXISTS "${${required}}")
        message(FATAL_ERROR "${required} does not exist: ${${required}}")
    endif()
endforeach()
if(NOT EXISTS "${EZ_GFX_INCLUDE_DIR}/ez_gfx_api.h")
    message(FATAL_ERROR "EZ_GFX_INCLUDE_DIR does not contain ez_gfx_api.h: ${EZ_GFX_INCLUDE_DIR}")
endif()

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(GLFW_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 7b6aead9fb88b3623e3b3725ebb42670cbe4c579
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glfw)

set(SHADER_SOURCE "${CMAKE_CURRENT_LIST_DIR}/textured_cube.slang")
get_filename_component(EZ_GFX_SHADER_ARTIFACT "${EZ_GFX_SHADER_ARTIFACT}" ABSOLUTE BASE_DIR "${CMAKE_BINARY_DIR}")
get_filename_component(SHADER_ARTIFACT_DIR "${EZ_GFX_SHADER_ARTIFACT}" DIRECTORY)

add_custom_command(
    OUTPUT "${EZ_GFX_SHADER_ARTIFACT}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_ARTIFACT_DIR}"
    COMMAND "${EZ_GFX_COMPILER}" "${SHADER_SOURCE}"
        --target spirv
        --target dxil
        --target metal
        --development
        --output "${EZ_GFX_SHADER_ARTIFACT}"
    DEPENDS "${SHADER_SOURCE}" "${EZ_GFX_SHARED_SHADER_MODULE}"
    WORKING_DIRECTORY "${EZ_GFX_ROOT}"
    VERBATIM
)
add_custom_target(textured_cube_shader DEPENDS "${EZ_GFX_SHADER_ARTIFACT}")

add_executable(textured_cube main.c)
add_dependencies(textured_cube textured_cube_shader)
set_target_properties(textured_cube PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
target_include_directories(textured_cube PRIVATE "${EZ_GFX_INCLUDE_DIR}")
target_link_libraries(textured_cube PRIVATE glfw "${EZ_GFX_FFI_LIBRARY}")

if(MSVC)
    target_compile_options(textured_cube PRIVATE /W4 /WX)
else()
    target_compile_options(textured_cube PRIVATE -Wall -Wextra -Werror -pedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(textured_cube_teardown_policy_tests teardown_policy_tests.c)
    set_target_properties(textured_cube_teardown_policy_tests PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
    target_include_directories(textured_cube_teardown_policy_tests PRIVATE "${EZ_GFX_INCLUDE_DIR}")
    if(MSVC)
        target_compile_options(textured_cube_teardown_policy_tests PRIVATE /W4 /WX)
    else()
        target_compile_options(textured_cube_teardown_policy_tests PRIVATE -Wall -Wextra -Werror -pedantic)
    endif()
    add_test(NAME textured_cube_teardown_policy COMMAND textured_cube_teardown_policy_tests)
endif()

add_custom_command(TARGET textured_cube POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${EZ_GFX_FFI_RUNTIME}" "$<TARGET_FILE_DIR:textured_cube>"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${EZ_GFX_SHADER_ARTIFACT}" "$<TARGET_FILE_DIR:textured_cube>/textured_cube.ezgfxshader"
    VERBATIM
)
