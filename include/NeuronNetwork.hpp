#pragma once
#include "Matrix.hpp"
#include "MultiMatrixT.hpp"
#include <memory>
#include <vector>
#include <functional>

typedef unsigned int uint;

class INeuronNetwork {
public:
  virtual std::vector<float> calculateOutputs(std::vector<float>) = 0;
  virtual Matrix calculateMultiOutputs(IMatrix&) = 0;
  
  static constexpr uint networkDepth = 100u;
private:
};

class NeuronNetwork : public INeuronNetwork {
public:
  NeuronNetwork();
  virtual std::vector<float> calculateOutputs(std::vector<float>);
  virtual Matrix calculateMultiOutputs(IMatrix&);

private:
  std::function<float(float)> activationFunction;
  std::shared_ptr<IMultiMatrixT<networkDepth>> weights;
};

