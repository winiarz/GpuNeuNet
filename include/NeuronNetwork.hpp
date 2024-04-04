#pragma once
#include "Matrix.hpp"
#include "MultiMatrixT.hpp"
#include <memory>
#include <vector>
#include <functional>
#include "ClTypedMemory.hpp"

typedef unsigned int uint;

class INeuronNetwork {
public:
  virtual std::vector<float> calculateOutputs(std::vector<float>) = 0;
  virtual Matrix calculateMultiOutputs(IMatrix&) = 0;

  virtual std::shared_ptr<ClTypedMemory<float>> copyToGpu() = 0;
  
  static constexpr uint networkDepth = 100u;
private:
};

class NeuronNetwork : public INeuronNetwork {
public:
  NeuronNetwork();
  virtual std::vector<float> calculateOutputs(std::vector<float>);
  virtual Matrix calculateMultiOutputs(IMatrix&);

  virtual std::shared_ptr<ClTypedMemory<float>> copyToGpu();

private:
  std::function<float(float)> activationFunction;
  std::shared_ptr<IMultiMatrixT<networkDepth>> weights;
};

