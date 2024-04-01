#pragma once
#include <vector>
#include <memory>
#include <functional>
#include "Matrix.hpp"

class INeuronLayer {
public:
  virtual std::vector<float> calculateOutputs(std::vector<float>) = 0;
  virtual Matrix calculateMultiOutputs(IMatrix&) = 0;
  virtual std::shared_ptr<ClTypedMemory<float>> copyToGpu() = 0;
};

class NeuronLayer : public INeuronLayer{
public:
  NeuronLayer();
  ~NeuronLayer();

  virtual std::vector<float> calculateOutputs(std::vector<float>);
  virtual Matrix calculateMultiOutputs(IMatrix&);
  virtual std::shared_ptr<ClTypedMemory<float>> copyToGpu();

private:
  std::function<float(float)> activationFunction;
  std::shared_ptr<IMatrix> weights;
};

