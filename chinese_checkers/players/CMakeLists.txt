file(GLOB players_source_files
        "*.cpp"
        "*.h")

add_library(players ${players_source_files})

target_link_libraries(players cc utils)