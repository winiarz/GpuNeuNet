#pragma once

#include <CL/cl.h>
#include <stl.hpp>
#include <memory>

typedef unsigned int uint;

class IClKernelCallStats;
class ClMemory;

class IClKernel {
public:
    virtual bool isSetUpSuccessfully()=0;
    virtual bool operator!()=0;
    virtual IClKernel& operator[](uint n)=0;
    virtual IClKernel& operator()(uint, ... )=0;
    virtual IClKernel& operator()(std::vector<ClMemory*>)=0;
    virtual IClKernel& operator()(std::vector<std::shared_ptr<ClMemory>>)=0;

    virtual ~IClKernel(){}
};

