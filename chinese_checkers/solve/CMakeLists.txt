file(GLOB solve_source_files
        "*.cpp"
        "*.h")

add_library(solve ${solve_source_files})

target_link_libraries(solve cc utils)