#pragma once
#include "MultiMatrixType.hpp"
#include <memory>
#include "ClTypedMemory.hpp"
#include "Matrix.hpp"

template<uint N>
class IMultiMatrixT {
public:
    virtual void set(float, uint, uint, uint) = 0;
    virtual float get(uint, uint, uint) = 0;
    Matrix getSingleMatrix(uint);
    virtual MultiMatrixType getType() = 0;

    void fillRandom();

    virtual float* getData() = 0;
    virtual std::shared_ptr<ClTypedMemory<float>> copyToGpu() = 0;
    bool operator==(IMultiMatrixT<N>&);

    static constexpr uint matrixSize = 256;
    static constexpr float epsilon = 0.001f;
};

template<uint N>
class MultiMatrixT_SeparateNormal : public IMultiMatrixT<N>{
public:
    virtual void set(float, uint, uint, uint);
    virtual float get(uint, uint, uint);
    virtual MultiMatrixType getType();

    virtual float* getData();
    virtual std::shared_ptr<ClTypedMemory<float>> copyToGpu();

private:
  float data[IMultiMatrixT<N>::matrixSize * IMultiMatrixT<N>::matrixSize*N];
};

template<uint N>
std::shared_ptr<IMultiMatrixT<N>> operator*(std::shared_ptr<IMultiMatrixT<N>>, std::shared_ptr<IMatrix>);

