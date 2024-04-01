#include "NeuronLayer.hpp"

NeuronLayer::NeuronLayer()
{
  weights = std::make_shared<Matrix>();
  weights->fillRandom();
}

std::vector<float> NeuronLayer::calculateOutputs(std::vector<float> inputs)
{
  std::vector<float> result;

  for(int i=0; i<IMatrix::matrixSize-1; i++)
  {
    float r = 0.0f;

    for(int j=0; j<IMatrix::matrixSize-1; j++)
    {
      r += inputs[j] * weights->get(i,j);
    }
    r+= weights->get(i, IMatrix::matrixSize-1);

    result.push_back(r);
  }

  return result;
}

