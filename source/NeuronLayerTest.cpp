#include "NeuronLayerTest.hpp"
#include "ClKernel.hpp"
#include "time.hpp"

std::shared_ptr<ClKernelFromSourceLoader> NeuronLayerTest::kernelLoader = nullptr;

NeuronLayerTest::NeuronLayerTest(std::string p_kernelSourceFilename, std::string p_kernelName) :
  kernelSourceFilename(p_kernelSourceFilename), kernelName(p_kernelName)
{
    neuronLayer = std::make_shared<NeuronLayer>();
    inputs = std::make_shared<Matrix>();

    if(kernelLoader == nullptr)
    {
        std::set<std::string> clIncludeDirs;
        kernelLoader = std::make_shared<ClKernelFromSourceLoader>(clIncludeDirs);
    }
}

void NeuronLayerTest::prepareTest()
{
    inputs->fillRandomInputs();
    kernel = kernelLoader->loadKernel(kernelSourceFilename, kernelName);

    gpuMemory.reserve(3);
    gpuMemory.emplace_back(inputs->copyToGpu());
    gpuMemory.emplace_back(neuronLayer->copyToGpu());
    gpuMemory.emplace_back(std::make_shared<ClTypedMemory<float>> (IMatrix::matrixSize*IMatrix::matrixSize));
}

bool NeuronLayerTest::performTest()
{
    prepareTest();

    measureTime(kernelName, [&](){(*kernel)[1u][256u](gpuMemory);});

    Matrix gpuResults;
    gpuMemory[2]->copyOut(gpuResults.getData(), 0, sizeof(float) * IMatrix::matrixSize * IMatrix::matrixSize );

    Matrix result = neuronLayer->calculateMultiOutputs(*inputs);

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

