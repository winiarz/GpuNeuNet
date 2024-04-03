#include <iostream>
#include "testLists.hpp"
#include "NeuronLayerTest.hpp"

int main()
{
    srand48(123456789);

    testSingleMatrixMultiply();
    testMultiMatrixMultiply();
    testSingleNeuronLayer();
}

