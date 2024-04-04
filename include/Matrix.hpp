#pragma once
#include <cstdint>
#include <ClTypedMemory.hpp>
#include <memory>

typedef unsigned int uint;

enum MatrixType {
MatrixType_normal,
MatrixType_swapped
};

class IMatrix {
public:
    virtual void set(float, uint, uint) = 0;
    virtual float get(uint, uint) const = 0;
    virtual MatrixType getType() = 0;

    void fillRandom();
    void fillRandomInputs();
    void print();
    void copyIn(const IMatrix&);

    virtual float* getData() = 0;
    bool operator==(IMatrix&);

    virtual std::shared_ptr<ClTypedMemory<float>> copyToGpu() = 0;

    static constexpr uint matrixSize = 256;
    static constexpr float epsilon = 0.001f;

protected:
    virtual uint getIdx(uint, uint) const = 0;
};

class Matrix : public IMatrix {
public:
    ~Matrix();

    virtual void set(float, uint, uint);
    virtual float get(uint, uint) const;
    virtual MatrixType getType();

    virtual float* getData();

    virtual std::shared_ptr<ClTypedMemory<float>> copyToGpu();

private:
    float m[matrixSize*matrixSize];

    virtual uint getIdx(uint, uint) const;
};

class MatrixSwapped : public IMatrix {
public:
    virtual void set(float, uint, uint);
    virtual float get(uint, uint) const;
    virtual MatrixType getType();

    virtual float* getData();
    virtual std::shared_ptr<ClTypedMemory<float>> copyToGpu();
private:
    float m[matrixSize*matrixSize];

    virtual uint getIdx(uint, uint) const;
};

Matrix operator*(IMatrix&, IMatrix&);

