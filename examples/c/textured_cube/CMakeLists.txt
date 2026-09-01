cmake_minimum_required(VERSION 3.24)
project(ez_gfx_textured_cube LANGUAGES C)

if(NOT WIN32)
    message(FATAL_ERROR "The current C presentation host is Win32-only; Metal requires an Objective-C CAMetalLayer host.")
endif()

set(EZ_GFX_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." CACHE PATH "ez_gfx_api2 repository root")
set(EZ_GFX_INCLUDE_DIR "${EZ_GFX_ROOT}/include" CACHE PATH "directory containing ez_gfx_api.h")
set(EZ_GFX_COMPILER "${EZ_GFX_ROOT}/target/debug/ez-gfx-compile.exe" CACHE FILEPATH "ez-gfx shader compiler")
set(EZ_GFX_FFI_LIBRARY "${EZ_GFX_ROOT}/target/debug/ez_gfx_ffi.dll.lib" CACHE FILEPATH "ez-gfx FFI import library")
set(EZ_GFX_FFI_DLL "${EZ_GFX_ROOT}/target/debug/ez_gfx_ffi.dll" CACHE FILEPATH "ez-gfx FFI runtime DLL")
set(EZ_GFX_SHADER_ARTIFACT "${EZ_GFX_ROOT}/target/c-examples/textured_cube/textured_cube.ezgfxshader" CACHE FILEPATH "generated shader artifact")
set(EZ_GFX_SHARED_SHADER_MODULE "${EZ_GFX_ROOT}/ez_gfx_api.slang" CACHE FILEPATH "shared ez-gfx shader module")

foreach(required EZ_GFX_INCLUDE_DIR EZ_GFX_COMPILER EZ_GFX_FFI_LIBRARY EZ_GFX_FFI_DLL EZ_GFX_SHARED_SHADER_MODULE)
    if(NOT EXISTS "${${required}}")
        message(FATAL_ERROR "${required} does not exist: ${${required}}")
    endif()
endforeach()
if(NOT EXISTS "${EZ_GFX_INCLUDE_DIR}/ez_gfx_api.h")
    message(FATAL_ERROR "EZ_GFX_INCLUDE_DIR does not contain ez_gfx_api.h: ${EZ_GFX_INCLUDE_DIR}")
endif()

set(SHADER_SOURCE "${CMAKE_CURRENT_LIST_DIR}/textured_cube.slang")
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
target_compile_features(textured_cube PRIVATE c_std_11)
target_include_directories(textured_cube PRIVATE "${EZ_GFX_INCLUDE_DIR}")
target_link_libraries(textured_cube PRIVATE "${EZ_GFX_FFI_LIBRARY}" user32)

if(MSVC)
    target_compile_options(textured_cube PRIVATE /W4 /WX)
else()
    target_compile_options(textured_cube PRIVATE -Wall -Wextra -Werror -pedantic)
endif()

add_custom_command(TARGET textured_cube POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${EZ_GFX_FFI_DLL}" "$<TARGET_FILE_DIR:textured_cube>"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${EZ_GFX_SHADER_ARTIFACT}" "$<TARGET_FILE_DIR:textured_cube>/textured_cube.ezgfxshader"
    VERBATIM
)
