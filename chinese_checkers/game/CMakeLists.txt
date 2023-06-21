file(GLOB game_source_files
        "*.cpp"
        "*.h")

add_library(game ${game_source_files})

target_link_libraries(game cc utils players)