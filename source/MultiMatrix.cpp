#include "MultiMatrix.hpp"
#include <stdlib.h>
#include <iostream>

bool IMultiMatrix::operator==(IMultiMatrix& other)
{
    for(uint a=0; a<matrixCount; a++)
      for(uint i=0; i<matrixSize; i++)
          for(uint j=0; j<matrixSize; j++)
              if(get(i,j,a) - other.get(i,j,a) > epsilon ||
                 get(i,j,a) - other.get(i,j,a) < -epsilon)
              {
                  std::cout << "MM " << i << " " << j << " " << a << " " << get(i,j,a) << " " << other.get(i,j,a) <<  std::endl;
                  return false;
              }

    return true;
}

void MultiMatrix_SeparateNormal::set(float value, uint x, uint y, uint n)
{
  data[x+matrixSize*(y + matrixSize*n)] = value;
}

float MultiMatrix_SeparateNormal::get(uint x, uint y, uint n)
{
  return data[x+matrixSize*(y + matrixSize*n)];
}

MultiMatrixType MultiMatrix_SeparateNormal::getType()
{
  return MultiMatrixType_separateNormal;
}

void IMultiMatrix::fillRandom()
{
  for(uint n=0; n<matrixCount; n++)
    for(uint y=0; y<matrixSize; y++)
      for(uint x=0; x<matrixSize; x++)
        set(drand48(), x, y, n);
}

float* MultiMatrix_SeparateNormal::getData()
{
  return data;
}

std::shared_ptr<IMultiMatrix> operator*(std::shared_ptr<IMultiMatrix> A, std::shared_ptr<IMatrix> B)
{
  std::shared_ptr<IMultiMatrix> result = std::make_shared<MultiMatrix_SeparateNormal>();

  for(uint a=0; a<IMultiMatrix::matrixCount; a++)
  {
    for(uint y=0; y<IMatrix::matrixSize; y++)
        for(uint x=0; x<IMatrix::matrixSize; x++)
        {
            float sum = 0.0f;

            for(uint i=0; i<IMatrix::matrixSize; i++)
            {
                sum += A->get(x,i,a) * B->get(i,y);
            }

            result->set(sum,x,y,a);
        }
  }

  return result;
}

std::shared_ptr<ClTypedMemory<float>> MultiMatrix_SeparateNormal::copyToGpu()
{
    auto result = std::make_shared<ClTypedMemory<float>> (matrixSize*matrixSize*matrixCount);
    result->copyIn(data, 0, matrixSize*matrixSize*matrixCount);
    return result;
}


