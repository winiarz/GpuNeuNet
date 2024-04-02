#pragma once
#include <memory>
#include "NeuronLayer.hpp"
#include "ClKernelFromSourceLoader.hpp"

class NeuronLayerTest {
public:
    NeuronLayerTest(std::string, std::string);

    bool performTest();
private:
    void prepareTest();
    
    std::string kernelSourceFilename, kernelName;
    std::shared_ptr<ClKernel> kernel;
    static std::shared_ptr<ClKernelFromSourceLoader> kernelLoader;
    std::shared_ptr<INeuronLayer> neuronLayer;
    std::shared_ptr<IMatrix> inputs;
    std::vector<std::shared_ptr<ClMemory>> gpuMemory;
};
