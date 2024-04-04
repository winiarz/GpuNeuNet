#include "NeuronNetwork.hpp"
#include <cmath>

NeuronNetwork::NeuronNetwork()
{
  weights = std::make_shared<MultiMatrixT_SeparateNormal<networkDepth>>();
  weights->fillRandom();

  activationFunction = [](float x){return 1.0f / (1.0f + exp(-x*0.01f));};
}

std::vector<float> NeuronNetwork::calculateOutputs(std::vector<float> inputs)
{
  std::vector<float> result, temp_inputs;

  std::copy(inputs.begin(), inputs.end(), std::back_inserter(temp_inputs));

  for(uint n=0; n<networkDepth; n++)
  {
    result.clear();

    for(int i=0; i<IMatrix::matrixSize-1; i++)
    {
      float r = 0.0f;

      for(int j=0; j<IMatrix::matrixSize-1; j++)
      {
        r += temp_inputs[j] * weights->get(j,i, n);
      }
      r+= weights->get(i, IMatrix::matrixSize-1, n);

      result.push_back(activationFunction(r));
    }
    result.push_back(1.0f);

    temp_inputs.clear();
    std::copy(result.begin(), result.end(), std::back_inserter(temp_inputs));
  }

  return result;
}

Matrix NeuronNetwork::calculateMultiOutputs(IMatrix& inputs)
{
  Matrix result, temp_inputs;
  temp_inputs.copyIn(inputs);
 
  for(uint n=0; n<networkDepth; n++)
  {
    Matrix layerWeights = weights->getSingleMatrix(n);
    result = temp_inputs * layerWeights;

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

    temp_inputs = result;
  }

  return result;
}

