# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/student/sortproject

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/student/sortproject/build

# Include any dependencies generated for this target.
include CMakeFiles/dotests.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dotests.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dotests.dir/flags.make

CMakeFiles/dotests.dir/tests/alltests.cpp.o: CMakeFiles/dotests.dir/flags.make
CMakeFiles/dotests.dir/tests/alltests.cpp.o: ../tests/alltests.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/student/sortproject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dotests.dir/tests/alltests.cpp.o"
	/opt/rh/devtoolset-7/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dotests.dir/tests/alltests.cpp.o -c /home/student/sortproject/tests/alltests.cpp

CMakeFiles/dotests.dir/tests/alltests.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dotests.dir/tests/alltests.cpp.i"
	/opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/student/sortproject/tests/alltests.cpp > CMakeFiles/dotests.dir/tests/alltests.cpp.i

CMakeFiles/dotests.dir/tests/alltests.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dotests.dir/tests/alltests.cpp.s"
	/opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/student/sortproject/tests/alltests.cpp -o CMakeFiles/dotests.dir/tests/alltests.cpp.s

CMakeFiles/dotests.dir/src/datageneration.cpp.o: CMakeFiles/dotests.dir/flags.make
CMakeFiles/dotests.dir/src/datageneration.cpp.o: ../src/datageneration.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/student/sortproject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/dotests.dir/src/datageneration.cpp.o"
	/opt/rh/devtoolset-7/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dotests.dir/src/datageneration.cpp.o -c /home/student/sortproject/src/datageneration.cpp

CMakeFiles/dotests.dir/src/datageneration.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dotests.dir/src/datageneration.cpp.i"
	/opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/student/sortproject/src/datageneration.cpp > CMakeFiles/dotests.dir/src/datageneration.cpp.i

CMakeFiles/dotests.dir/src/datageneration.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dotests.dir/src/datageneration.cpp.s"
	/opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/student/sortproject/src/datageneration.cpp -o CMakeFiles/dotests.dir/src/datageneration.cpp.s

# Object files for target dotests
dotests_OBJECTS = \
"CMakeFiles/dotests.dir/tests/alltests.cpp.o" \
"CMakeFiles/dotests.dir/src/datageneration.cpp.o"

# External object files for target dotests
dotests_EXTERNAL_OBJECTS =

bin/dotests: CMakeFiles/dotests.dir/tests/alltests.cpp.o
bin/dotests: CMakeFiles/dotests.dir/src/datageneration.cpp.o
bin/dotests: CMakeFiles/dotests.dir/build.make
bin/dotests: lib/libutils.a
bin/dotests: lib/libstudent_solution.a
bin/dotests: /usr/lib64/libgtest.a
bin/dotests: /usr/lib64/libgtest_main.a
bin/dotests: /usr/lib64/mpich-3.2/lib/libmpicxx.so
bin/dotests: /usr/lib64/mpich-3.2/lib/libmpi.so
bin/dotests: CMakeFiles/dotests.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/student/sortproject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable bin/dotests"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dotests.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dotests.dir/build: bin/dotests

.PHONY : CMakeFiles/dotests.dir/build

CMakeFiles/dotests.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dotests.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dotests.dir/clean

CMakeFiles/dotests.dir/depend:
	cd /home/student/sortproject/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/student/sortproject /home/student/sortproject /home/student/sortproject/build /home/student/sortproject/build /home/student/sortproject/build/CMakeFiles/dotests.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dotests.dir/depend

