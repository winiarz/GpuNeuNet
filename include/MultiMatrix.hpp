#pragma once
#include <memory>
#include "Matrix.hpp"
#include "MultiMatrixType.hpp"

typedef unsigned int uint;

class IMultiMatrix {
public:
    virtual void set(float, uint, uint, uint) = 0;
    virtual float get(uint, uint, uint) = 0;
    virtual MultiMatrixType getType() = 0;

    void fillRandom();

    virtual float* getData() = 0;
    virtual std::shared_ptr<ClTypedMemory<float>> copyToGpu() = 0;
    bool operator==(IMultiMatrix&);

    static constexpr uint matrixSize = 256;
    static constexpr uint matrixCount = 4;
    static constexpr float epsilon = 0.001f;
};


class MultiMatrix_SeparateNormal : public IMultiMatrix {
public:
    virtual void set(float, uint, uint, uint);
    virtual float get(uint, uint, uint);
    virtual MultiMatrixType getType();
    virtual float* getData();
    virtual std::shared_ptr<ClTypedMemory<float>> copyToGpu();

private:
  float data[matrixSize*matrixSize*matrixCount];
};

class MultiMatrix_CombinedNormal : public IMultiMatrix {
public:
    virtual void set(float, uint, uint, uint);
    virtual float get(uint, uint, uint);
    virtual MultiMatrixType getType();
    virtual float* getData();
    virtual std::shared_ptr<ClTypedMemory<float>> copyToGpu();

private:
  float data[matrixSize*matrixSize*matrixCount];
};

std::shared_ptr<IMultiMatrix> operator*(std::shared_ptr<IMultiMatrix>, std::shared_ptr<IMatrix>);

