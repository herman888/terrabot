cmake_minimum_required(VERSION 3.10)
project(TerrabotMainframe)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(terrabot_mainframe main.cpp)

# Platform-specific settings
if(WIN32)
    # Windows: Link Winsock2
    target_link_libraries(terrabot_mainframe ws2_32)
else()
    # Linux/Unix: Link pthread
    find_package(Threads REQUIRED)
    target_link_libraries(terrabot_mainframe Threads::Threads)
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(terrabot_mainframe PRIVATE /W4)
else()
    target_compile_options(terrabot_mainframe PRIVATE -Wall -Wextra -pedantic)
endif()

# Set output directory
set_target_properties(terrabot_mainframe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

message(STATUS "Configuring Terrabot Mainframe")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
