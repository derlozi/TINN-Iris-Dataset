# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.10

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2018.1.2\bin\cmake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2018.1.2\bin\cmake\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\loren\Documents\Programmieren\C\FirstTINN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\loren\Documents\Programmieren\C\FirstTINN\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/FirstTINN.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/FirstTINN.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FirstTINN.dir/flags.make

CMakeFiles/FirstTINN.dir/main.c.obj: CMakeFiles/FirstTINN.dir/flags.make
CMakeFiles/FirstTINN.dir/main.c.obj: CMakeFiles/FirstTINN.dir/includes_C.rsp
CMakeFiles/FirstTINN.dir/main.c.obj: ../main.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\loren\Documents\Programmieren\C\FirstTINN\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/FirstTINN.dir/main.c.obj"
	C:\TDM-GCC-64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\FirstTINN.dir\main.c.obj   -c C:\Users\loren\Documents\Programmieren\C\FirstTINN\main.c

CMakeFiles/FirstTINN.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/FirstTINN.dir/main.c.i"
	C:\TDM-GCC-64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\loren\Documents\Programmieren\C\FirstTINN\main.c > CMakeFiles\FirstTINN.dir\main.c.i

CMakeFiles/FirstTINN.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/FirstTINN.dir/main.c.s"
	C:\TDM-GCC-64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\loren\Documents\Programmieren\C\FirstTINN\main.c -o CMakeFiles\FirstTINN.dir\main.c.s

CMakeFiles/FirstTINN.dir/main.c.obj.requires:

.PHONY : CMakeFiles/FirstTINN.dir/main.c.obj.requires

CMakeFiles/FirstTINN.dir/main.c.obj.provides: CMakeFiles/FirstTINN.dir/main.c.obj.requires
	$(MAKE) -f CMakeFiles\FirstTINN.dir\build.make CMakeFiles/FirstTINN.dir/main.c.obj.provides.build
.PHONY : CMakeFiles/FirstTINN.dir/main.c.obj.provides

CMakeFiles/FirstTINN.dir/main.c.obj.provides.build: CMakeFiles/FirstTINN.dir/main.c.obj


# Object files for target FirstTINN
FirstTINN_OBJECTS = \
"CMakeFiles/FirstTINN.dir/main.c.obj"

# External object files for target FirstTINN
FirstTINN_EXTERNAL_OBJECTS =

FirstTINN.exe: CMakeFiles/FirstTINN.dir/main.c.obj
FirstTINN.exe: CMakeFiles/FirstTINN.dir/build.make
FirstTINN.exe: tinn-master/libTinn.a
FirstTINN.exe: CMakeFiles/FirstTINN.dir/linklibs.rsp
FirstTINN.exe: CMakeFiles/FirstTINN.dir/objects1.rsp
FirstTINN.exe: CMakeFiles/FirstTINN.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\loren\Documents\Programmieren\C\FirstTINN\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable FirstTINN.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\FirstTINN.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FirstTINN.dir/build: FirstTINN.exe

.PHONY : CMakeFiles/FirstTINN.dir/build

CMakeFiles/FirstTINN.dir/requires: CMakeFiles/FirstTINN.dir/main.c.obj.requires

.PHONY : CMakeFiles/FirstTINN.dir/requires

CMakeFiles/FirstTINN.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\FirstTINN.dir\cmake_clean.cmake
.PHONY : CMakeFiles/FirstTINN.dir/clean

CMakeFiles/FirstTINN.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\loren\Documents\Programmieren\C\FirstTINN C:\Users\loren\Documents\Programmieren\C\FirstTINN C:\Users\loren\Documents\Programmieren\C\FirstTINN\cmake-build-debug C:\Users\loren\Documents\Programmieren\C\FirstTINN\cmake-build-debug C:\Users\loren\Documents\Programmieren\C\FirstTINN\cmake-build-debug\CMakeFiles\FirstTINN.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FirstTINN.dir/depend

