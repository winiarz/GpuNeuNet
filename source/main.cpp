#include <iostream>
#include "testLists.hpp"
#include "MultiMatrixT.hpp"

int main()
{
    srand48(123456789);

    testSingleMatrixMultiply();
    testMultiMatrixMultiply();
    testSingleNeuronLayer();

    MultiMatrixT_SeparateNormal<10> mm;
}

