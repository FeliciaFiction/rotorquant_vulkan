cmake_minimum_required(VERSION 3.20)
project(turboquant-vulkan-shaders C)

find_program(GLSLC_EXECUTABLE NAMES glslc)
if (NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found on PATH")
endif()

set(SHADER_SOURCES
    qjl_quant.comp
    qjl_score.comp
    qjl_gqa_score.comp
)

set(SPIRV_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/spv")
file(MAKE_DIRECTORY "${SPIRV_OUTPUT_DIR}")

foreach(SHADER_FILE IN LISTS SHADER_SOURCES)
    get_filename_component(SHADER_NAME "${SHADER_FILE}" NAME)
    set(OUTPUT_FILE "${SPIRV_OUTPUT_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT "${OUTPUT_FILE}"
        COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.3
                -o "${OUTPUT_FILE}" "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_FILE}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_FILE}"
        COMMENT "Compiling ${SHADER_NAME} to SPIR-V"
    )
    list(APPEND SPIRV_TARGETS "${OUTPUT_FILE}")
endforeach()

add_custom_target(compile_shaders ALL DEPENDS ${SPIRV_TARGETS})

