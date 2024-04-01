#pragma once
#include <vector>
#include <memory>
#include "Matrix.hpp"

class NeuronLayer {
public:
  NeuronLayer();

  std::vector<float> calculateOutputs(std::vector<float>);
private:
  std::shared_ptr<IMatrix> weights;
};

