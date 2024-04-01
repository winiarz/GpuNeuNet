#include "NeuronLayer.hpp"
#include <cmath>
#include <iostream>

NeuronLayer::NeuronLayer()
{
  weights = std::make_shared<Matrix>();
  weights->fillRandom();

  activationFunction = [](float x){return 1.0f / (1.0f + exp(-x*0.01f));};
}

NeuronLayer::~NeuronLayer()
{
}

std::vector<float> NeuronLayer::calculateOutputs(std::vector<float> inputs)
{
  std::vector<float> result;

  for(int i=0; i<IMatrix::matrixSize-1; i++)
  {
    float r = 0.0f;

    for(int j=0; j<IMatrix::matrixSize-1; j++)
    {
      r += inputs[j] * weights->get(j,i);
    }
    r+= weights->get(i, IMatrix::matrixSize-1);

    result.push_back(activationFunction(r));
  }

  return result;
}

Matrix NeuronLayer::calculateMultiOutputs(IMatrix& inputs)
{
  Matrix result;
  
  result = inputs * (*weights);

  for(int i=0; i<IMatrix::matrixSize; i++)
  {
    result.set(1.0f, i, IMatrix::matrixSize-1);

    for(int j=0; j<IMatrix::matrixSize-1; j++)
    {
      float temp = result.get(i,j);
      float temp2 = activationFunction(temp);
      result.set(temp2, i, j);
    }
  }

  return result;
}

std::shared_ptr<ClTypedMemory<float>> NeuronLayer::copyToGpu()
{
  return weights->copyToGpu();
}

