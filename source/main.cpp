#include <iostream>
#include "testLists.hpp"
#include "NeuronNetworkTest.hpp"

int main()
{
    srand48(123456789);

    testSingleMatrixMultiply();
    testMultiMatrixMultiply();
    testSingleNeuronLayer();

    NeuronNetworkTest test("kernel/neuron_network.cl", "neuron_network_simple");
    test.performTest();
}

