#include <iostream>
#include "testLists.hpp"
#include "NeuronLayer.hpp"
#include "ClKernelFromSourceLoader.hpp"
#include "time.hpp"
#include "ClKernel.hpp"

int main()
{
    srand48(123456789);

    //testSingleMatrixMultiply();
    //testMultiMatrixMultiply();

    std::vector<std::shared_ptr<ClMemory>> gpuData;

    Matrix inputs;
    inputs.fillRandomInputs();
    gpuData.push_back( inputs.copyToGpu() );

    NeuronLayer neuronLayer;
    gpuData.push_back( neuronLayer.copyToGpu() );
    Matrix result = neuronLayer.calculateMultiOutputs(inputs);

    gpuData.push_back( std::make_shared<ClTypedMemory<float>> (IMatrix::matrixSize * IMatrix::matrixSize));

    std::set<std::string> clIncludeDirs;
    ClKernelFromSourceLoader kernelLoader(clIncludeDirs);

    auto kernel = kernelLoader.loadKernel("kernel/single_neur_layer.cl", "neur_layer_simple");

    measureTime("neur_layer_simple", [&](){(*kernel)[1u][256u](gpuData);});

    Matrix gpuResults;
    gpuData[2]->copyOut(gpuResults.getData(), 0, sizeof(float) * IMatrix::matrixSize * IMatrix::matrixSize );

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

