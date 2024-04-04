#include "NeuronNetworkTest.hpp"
#include "ClKernel.hpp"
#include "time.hpp"

std::shared_ptr<ClKernelFromSourceLoader> NeuronNetworkTest::kernelLoader = nullptr;

NeuronNetworkTest::NeuronNetworkTest(std::string p_filename, std::string p_kernelName) :
  kernelSourceFilename(p_filename), kernelName(p_kernelName)
{
    neuronNetwork = std::make_shared<NeuronNetwork>();
    inputs = std::make_shared<Matrix>();

    if(kernelLoader == nullptr)
    {
        std::set<std::string> clIncludeDirs;
        kernelLoader = std::make_shared<ClKernelFromSourceLoader>(clIncludeDirs);
    }

}

void NeuronNetworkTest::prepareTest()
{
    inputs->fillRandomInputs();
    kernel = kernelLoader->loadKernel(kernelSourceFilename, kernelName);

    gpuMemory.reserve(3);
    gpuMemory.emplace_back(inputs->copyToGpu());
    gpuMemory.emplace_back(neuronNetwork->copyToGpu());
    gpuMemory.emplace_back(std::make_shared<ClTypedMemory<float>> (IMatrix::matrixSize*IMatrix::matrixSize));
}

bool NeuronNetworkTest::performTest()
{
    prepareTest();

    measureTime(kernelName, [&](){(*kernel)[1u][256u](gpuMemory);});

    Matrix gpuResults;
    gpuMemory[0]->copyOut(gpuResults.getData(), 0, sizeof(float) * IMatrix::matrixSize * IMatrix::matrixSize );

    Matrix result = neuronNetwork->calculateMultiOutputs(*inputs);

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

