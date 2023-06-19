file(GLOB cc_source_files
        "*.cpp"
        "*.h")

add_library(cc ${cc_source_files})

target_link_libraries(cc utils)