#include "Matrix.hpp"
#include <stdlib.h>


void MatrixSwapped::set(float value, uint x, uint y)
{
    m[getIdx(x,y)] = value;
}

float MatrixSwapped::get(uint x, uint y) const
{
    return m[getIdx(x,y)];
}

std::shared_ptr<ClTypedMemory<float>> MatrixSwapped::copyToGpu()
{
    auto result = std::make_shared<ClTypedMemory<float>> (matrixSize*matrixSize);
    result->copyIn(m, 0, matrixSize*matrixSize);
    return result;
}

float* MatrixSwapped::getData()
{
    return m;
}

uint MatrixSwapped::getIdx(uint x, uint y) const
{
    return y + matrixSize*x;
}

MatrixType MatrixSwapped::getType()
{
    return MatrixType_swapped;
}

