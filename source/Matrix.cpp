#include "Matrix.hpp"
#include <stdlib.h>
#include <iostream>

bool IMatrix::operator==(IMatrix& other)
{
    for(uint i=0; i<matrixSize; i++)
        for(uint j=0; j<matrixSize; j++)
            if(get(i,j) - other.get(i,j) > epsilon ||
               get(i,j) - other.get(i,j) < -epsilon)
            {
                std::cout << "M " << i << " " << j << " " << get(i,j) << " " << other.get(i,j) <<  std::endl;
                return false;
            }

    return true;
}

Matrix::~Matrix()
{
}


void Matrix::set(float value, uint x, uint y)
{
    m[getIdx(x,y)] = value;
}

float Matrix::get(uint x, uint y) const
{
    return m[getIdx(x,y)];
}

void IMatrix::fillRandom()
{
    for(uint x=0; x<matrixSize; x++)
        for(uint y=0; y<matrixSize; y++)
            set(drand48(), x, y);
}

void IMatrix::fillRandomInputs()
{
    for(uint x=0; x<matrixSize; x++)
    {
        for(uint y=0; y<matrixSize-1; y++)
            set(drand48(), x, y);
 
        set(1.0f, x, matrixSize-1);
    }
}

void IMatrix::copyIn(const IMatrix& source)
{
  for(uint x=0; x<matrixSize; x++)
    for(uint y=0; y<matrixSize; y++)
      set( source.get(x,y), x, y);
}

float* Matrix::getData()
{
    return m;
}

std::shared_ptr<ClTypedMemory<float>> Matrix::copyToGpu()
{
    auto result = std::make_shared<ClTypedMemory<float>> (matrixSize*matrixSize);
    result->copyIn(m, 0, matrixSize*matrixSize);
    return result;
}

Matrix operator*(IMatrix& A, IMatrix& B)
{
    Matrix result;

    for(uint y=0; y<IMatrix::matrixSize; y++)
        for(uint x=0; x<IMatrix::matrixSize; x++)
        {
            float sum = 0.0f;

            for(uint i=0; i<IMatrix::matrixSize; i++)
            {
                sum += A.get(x,i) * B.get(i,y);
            }

            result.set(sum,x,y);
        }

    return result;
}

uint Matrix::getIdx(uint x, uint y) const
{
    return x + matrixSize*y;
}

void IMatrix::print()
{
    for(uint y=0; y<= matrixSize; y++)
    {
        for(uint x=0; x<=matrixSize; x++)
            std::cout << get(x,y) << " ";
        std::cout << std::endl;
    }
}

MatrixType Matrix::getType()
{
    return MatrixType_normal;
}

