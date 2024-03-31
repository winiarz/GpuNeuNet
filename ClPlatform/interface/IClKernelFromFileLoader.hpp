#pragma once

#include <memory>
#include "stl.hpp"

class ClKernel;

class IClKernelFromFileLoader
{
public:
    virtual std::shared_ptr<ClKernel> loadKernel(std::string filename) = 0;
};

