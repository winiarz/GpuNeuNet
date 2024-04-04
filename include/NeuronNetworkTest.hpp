#pragma once
#include "NeuronNetwork.hpp"
#include "ClKernelFromSourceLoader.hpp"
#include <string>

class NeuronNetworkTest {
public:
    NeuronNetworkTest(std::string, std::string);

    bool performTest();
private:
    void prepareTest();

    std::string kernelSourceFilename, kernelName;
    std::shared_ptr<ClKernel> kernel;
    static std::shared_ptr<ClKernelFromSourceLoader> kernelLoader;
    std::shared_ptr<INeuronNetwork> neuronNetwork;
    std::shared_ptr<IMatrix> inputs;
    std::vector<std::shared_ptr<ClMemory>> gpuMemory;
};

