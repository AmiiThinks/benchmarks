file(GLOB utils_source_files
        "*.cpp"
        "*.h")

add_library(utils ${utils_source_files})