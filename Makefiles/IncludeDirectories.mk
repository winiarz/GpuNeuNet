

#C++ include directories
ClPlatformInclude= -I$(ClPlatform)/include -I$(ClPlatform)/interface -I$(ClPlatform)/mock
CommonInclude= -I$(Common)/include -I$(Common)/interface -I$(Common)/mock

#Mock includes
MockIncludes= -I$(ClPlatform)/mock

AllInclude=$(ClPlatformInclude) $(CommonInclude) -I clinclude -I include



#OpenCL include folders
ClInclude= -I $(ProjectRoot)/clinclude
