#include <iostream>
#include "testLists.hpp"
#include "NeuronNetwork.hpp"
#include "ClKernelFromSourceLoader.hpp"
#include <vector>
#include "ClMemory.hpp"
#include "time.hpp"
#include "ClKernel.hpp"

int main()
{
    srand48(123456789);

    testSingleMatrixMultiply();
    testMultiMatrixMultiply();
    testSingleNeuronLayer();

    NeuronNetwork nn;

    Matrix inputs;
    inputs.fillRandomInputs();

    Matrix result;

    measureTime("CPU Neuron Network", [&](){ result = nn.calculateMultiOutputs(inputs);});

    std::set<std::string> clIncludeDirs;
    auto kernelLoader = std::make_shared<ClKernelFromSourceLoader>(clIncludeDirs);

    auto kernel = kernelLoader->loadKernel("kernel/neuron_network.cl", "neuron_network_simple");
    
    std::vector<std::shared_ptr<ClMemory>> gpuMemory;
    gpuMemory.emplace_back(inputs.copyToGpu());
    gpuMemory.emplace_back(nn.copyToGpu());
    gpuMemory.emplace_back(std::make_shared<ClTypedMemory<float>> (IMatrix::matrixSize*IMatrix::matrixSize));

    measureTime("neuron_network_simple", [&](){(*kernel)[1u][256u](gpuMemory);});

    Matrix gpuResults;
    gpuMemory[0]->copyOut(gpuResults.getData(), 0, sizeof(float) * IMatrix::matrixSize * IMatrix::matrixSize );

    if(result == gpuResults)
    {
        std::cout << "Correct!" << std::endl;
        return true;
    }
    else
    {
        std::cout << "Something wrong" << std::endl;
        return false;
    }

}

