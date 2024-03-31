#pragma once

#include <memory>
#include "stl.hpp"
class IClSingleImplementationKernel;

class IClKernelSaver
{
public:
    virtual void saveKernel( std::shared_ptr<IClSingleImplementationKernel> kernel, std::string filename ) = 0;
    virtual void saveKernel( std::shared_ptr<IClSingleImplementationKernel> kernel, FILE* file ) = 0;
};

