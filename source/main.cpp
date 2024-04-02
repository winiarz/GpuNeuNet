#include <iostream>
#include "testLists.hpp"
#include "NeuronLayerTest.hpp"

int main()
{
    srand48(123456789);

    testSingleMatrixMultiply();
    //testMultiMatrixMultiply();

    NeuronLayerTest test1("kernel/single_neur_layer.cl", "neur_layer_simple");
    test1.performTest();

    NeuronLayerTest test2("kernel/single_neur_layer.cl", "neur_layer_opt");
    test2.performTest();
}

