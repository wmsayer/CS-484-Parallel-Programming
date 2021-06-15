cmake_minimum_required (VERSION 3.5)
project(mp3 VERSION 1.0.0 LANGUAGES C CXX)

# Location of additional cmake modules
#
add_definitions(-DSP19)#TODO: Remove this and the dependent code
#add_definitions(-DMEASURE_SORT_TIME)#TODO: Add this for next semester.

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake
    )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_DATA_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/data)

include(FindOpenMP)
if(OPENMP_FOUND)
		message("Found OpenMP, building parallel part.")

        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

else(OPENMP_FOUND)
		message(FATAL_ERROR "ERROR: OpenMP not found.")
endif(OPENMP_FOUND)

include(FindMPI)
if(MPI_CXX_FOUND)
	message("Found MPI")
else(MPI_CXX_FOUND)
	message(FATAL_ERROR "Could not find MPI")
endif(MPI_CXX_FOUND)


include_directories(${MPI_CXX_INCLUDE_PATH})
include_directories("${PROJECT_SOURCE_DIR}/utils/include")


add_library(utils "utils/src/databasics.cpp" "utils/include/databasics.h")
add_library(orderedio "utils/src/orderedio.c" "utils/include/orderedio.h")

add_library(student_solution "src/solution.cpp"
	"src/basic_defs.h"
	"src/solution.h"
)

add_executable(sorter
		"src/main_sort.cpp"
		"src/basic_defs.h"
		"src/solution.h"
		"src/solution.cpp"
		"src/datageneration.h"
		"src/datageneration.cpp"
)
target_include_directories(sorter PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(sorter utils orderedio m ${MPI_CXX_LIBRARIES})

if(MPI_COMPILE_FLAGS)
  set_target_properties(sorter PROPERTIES
  COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(sorter PROPERTIES
  LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
endif()



#Tests and grading
include(CTest)
include(FindGTest)

if(GTEST_FOUND)
include(GoogleTest)


add_executable(dotests "tests/alltests.cpp"
						"tests/gtest-mpi-listener.hpp"
						"src/datageneration.h"
						"src/datageneration.cpp")
include_directories(dotests "src" "utils/include")
target_link_libraries(dotests utils m student_solution)
include_directories(dotests ${GTEST_INCLUDE_DIRS})
target_link_libraries(dotests ${GTEST_BOTH_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(utiltests "tests/utiltests.cpp" "tests/gtest-mpi-listener.hpp")
include_directories(utiltests "src" "utils/include")
target_link_libraries(utiltests utils)
include_directories(utiltests ${GTEST_INCLUDE_DIRS})
target_link_libraries(utiltests ${GTEST_BOTH_LIBRARIES} ${MPI_CXX_LIBRARIES})

if(DEFINED GRADING)
	#Don't compile student tests at grading time.
message(STATUS "BUILDING FOR GRADING: ${GRADING}.")
add_definitions(-DGRADING="${GRADING}")
else(DEFINED GRADING)
	#Student tests only get compiled when _not_ grading.
add_definitions(-DGRADING="${GRADING}")
add_executable(studenttests "tests/student_tests.cpp"
						"tests/gtest-mpi-listener.hpp"
						"src/datageneration.h"
						"src/datageneration.cpp")
include_directories(studenttests "src" "utils/include")
target_link_libraries(studenttests utils m student_solution)
include_directories(studenttests ${GTEST_INCLUDE_DIRS})
target_link_libraries(studenttests ${GTEST_BOTH_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif(DEFINED GRADING)

#copy the data directory to the build directory
#add_custom_command(TARGET dotests POST_BUILD
#	COMMAND ${CMAKE_COMMAND} -E copy_directory
#	${CMAKE_SOURCE_DIR}/data ${CMAKE_DATA_OUTPUT_DIRECTORY})

#gtest_discover_tests(dotests)
else(GTEST_FOUND)
	message("DANGER: Could not find GoogleTest")
endif(GTEST_FOUND)

#Benchmarking
include(FindGBench)
if(GBENCH_FOUND)
#add_executable(dobench "tests/allbench.cpp" "gs_parallel/gsparallel.cpp" "gs_serial/gsserial.cpp")
#include_directories(dobench "gstools/include" "gs_serial" "gs_parallel")
#target_link_libraries(dobench gstools)
#include_directories(dobench ${GBENCH_INCLUDE_DIRS})
#target_link_libraries(dobench ${GBENCH_BOTH_LIBRARIES})
else(GBENCH_FOUND)
	message("DANGER: Could not find GoogleBenchmark")
endif(GBENCH_FOUND)
