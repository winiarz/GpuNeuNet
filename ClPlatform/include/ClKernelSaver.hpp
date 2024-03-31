#pragma once

#include <CL/cl.h>
#include "IClKernelSaver.hpp"

class ClKernelSaver : public IClKernelSaver
{
public:
    virtual void saveKernel( std::shared_ptr<IClSingleImplementationKernel> kernel, std::string filename );
    virtual void saveKernel( std::shared_ptr<IClSingleImplementationKernel> kernel, FILE* file );
private:
    cl_int getDeviceCount( std::shared_ptr<IClSingleImplementationKernel> kernel);
    void getDevices( cl_device_id devices[], std::shared_ptr<IClSingleImplementationKernel> kernel );
    void getBinarySizes( size_t binarySizes[], std::shared_ptr<IClSingleImplementationKernel> kernel );

    void getProgramBinaries( size_t deviceCount,
                             size_t binarySizes[],
                             unsigned char *programBinaries[],
                             std::shared_ptr<IClSingleImplementationKernel> kernel);

    void deleteProgramBinaries( size_t deviceCount, unsigned char *programBinaries[] );
    size_t getDeviceIdx( size_t deviceCount, cl_device_id devices[] );
    void saveBinaryToFile( size_t binarySize, unsigned char *programBinary, FILE* file);
    FILE* openFile( std::string );
};

