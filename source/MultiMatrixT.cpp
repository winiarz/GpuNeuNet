#include "MultiMatrixT.hpp"
#include <iostream>

template<uint N>
bool IMultiMatrixT<N>::operator==(IMultiMatrixT<N>& other)
{
    for(uint a=0; a<N; a++)
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

template<uint N>
void IMultiMatrixT<N>::fillRandom()
{
  for(uint n=0; n<N; n++)
    for(uint y=0; y<IMultiMatrixT<N>::matrixSize; y++)
      for(uint x=0; x<IMultiMatrixT<N>::matrixSize; x++)
        set(drand48(), x, y, n);
}

template<uint N>
void MultiMatrixT_SeparateNormal<N>::set(float value, uint x, uint y, uint n)
{
  data[x+IMultiMatrixT<N>::matrixSize * (y + IMultiMatrixT<N>::matrixSize*n)] = value;
}

template<uint N>
float MultiMatrixT_SeparateNormal<N>::get(uint x, uint y, uint n)
{
  return data[x+IMultiMatrixT<N>::matrixSize*(y + IMultiMatrixT<N>::matrixSize*n)];
}

template<uint N>
MultiMatrixType MultiMatrixT_SeparateNormal<N>::getType()
{
  return MultiMatrixType_separateNormal;
}

template<uint N>
float* MultiMatrixT_SeparateNormal<N>::getData()
{
  return data;
}

template<uint N>
std::shared_ptr<ClTypedMemory<float>> MultiMatrixT_SeparateNormal<N>::copyToGpu()
{
    auto result = std::make_shared<ClTypedMemory<float>> (IMultiMatrixT<N>::matrixSize*IMultiMatrixT<N>::matrixSize*N);
    result->copyIn(data, 0, IMultiMatrixT<N>::matrixSize*IMultiMatrixT<N>::matrixSize*N);
    return result;
}

template<uint N>
std::shared_ptr<IMultiMatrixT<N>> operator*(std::shared_ptr<IMultiMatrixT<N>> A, std::shared_ptr<IMatrix> B)
{
  std::shared_ptr<IMultiMatrixT<N>> result = std::make_shared<MultiMatrixT_SeparateNormal<N>>();

  for(uint a=0; a<N; a++)
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

