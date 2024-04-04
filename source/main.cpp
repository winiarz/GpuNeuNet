#include <iostream>
#include "testLists.hpp"
#include "NeuronNetwork.hpp"

int main()
{
    srand48(123456789);

    testSingleMatrixMultiply();
    testMultiMatrixMultiply();
    testSingleNeuronLayer();

    NeuronNetwork nn;

    Matrix inputs;
    inputs.fillRandomInputs();

    Matrix outputs = nn.calculateMultiOutputs(inputs);
}

