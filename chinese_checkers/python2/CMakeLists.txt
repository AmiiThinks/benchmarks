file(GLOB dl_source_files
        "*.cpp"
        "*.h")

add_library(dl ${dl_source_files})

target_link_libraries(dl)
