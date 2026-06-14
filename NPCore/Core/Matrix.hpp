#ifndef NPCORE_CORE_MATRIX_HPP
#define NPCORE_CORE_MATRIX_HPP

#pragma once

#include <cmath>
#include <vector>
#include <thread>
#include <iomanip>
#include <string>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

#include "Assert.h"
#include "Size.h"
#include "Export.h"
#include "GEMM.h"
#include "CudaBridge.h"

// =================================[Device]================================
enum class Device { CPU, GPU };

// =================================[Matrix 矩阵类]================================
template <typename T>
class Matrix {
    template<typename U>
    friend void GramSchmidtOrthogonalization(Matrix<U>& mat);

public:
    int row{}, col{}, channel{};

private:
    T *data{};
    Device device_ = Device::CPU;
    bool owns_data = true;
    int currentIndex{};
    // ==============================[私有变量定义结束]==============================

public:
    // ===================================[构造函数]=======================================
    Matrix() = default;

    Matrix(T *data, int row, int col, int channel) :
            row(row), col(col), channel(channel), data(data), owns_data(false), currentIndex(0) {}

    Matrix(int row, int col) : row(row), col(col), channel(1), owns_data(true), currentIndex(0) {
        data = new T[row * col]();
    }

    Matrix(int row, int col, int channel) : row(row), col(col), channel(channel), owns_data(true), currentIndex(0) {
        data = new T[row * col * channel]();
    }
    // ===================================[构造函数定义结束]=======================================

    // 析构函数: 释放自有内存
    ~Matrix() {
        if (owns_data && data) {
            delete[] data;
            data = nullptr;
        }
    }

    // 拷贝构造: 深拷贝
    Matrix(const Matrix<T>& other) :
        row(other.row), col(other.col), channel(other.channel),
        owns_data(true), currentIndex(other.currentIndex) {
        const int total = row * col * channel;
        data = new T[total];
        for (int i = 0; i < total; ++i) data[i] = other.data[i];
    }

    // 移动构造: 转移所有权
    Matrix(Matrix<T>&& other) noexcept :
        row(other.row), col(other.col), channel(other.channel), data(other.data),
        owns_data(other.owns_data), currentIndex(other.currentIndex) {
        other.data = nullptr;
        other.owns_data = false;
    }

    // 拷贝赋值
    Matrix& operator=(const Matrix<T>& other) {
        if (this != &other) {
            if (owns_data && data) delete[] data;
            row = other.row;
            col = other.col;
            channel = other.channel;
            owns_data = true;
            currentIndex = other.currentIndex;
            const int total = row * col * channel;
            data = new T[total];
            for (int i = 0; i < total; ++i) data[i] = other.data[i];
        }
        return *this;
    }

    // 移动赋值
    Matrix& operator=(Matrix<T>&& other) noexcept {
        if (this != &other) {
            if (owns_data && data) delete[] data;
            data = other.data;
            row = other.row;
            col = other.col;
            channel = other.channel;
            owns_data = other.owns_data;
            currentIndex = other.currentIndex;
            other.data = nullptr;
            other.owns_data = false;
        }
        return *this;
    }

    // ===================================[操作符声明]=======================================

    Matrix<T> operator*(Matrix<T> mat2) const;
    Matrix<T> operator*(T value) const;
    Matrix<T> operator&(Matrix<T> mat2) const;

    Matrix<T>& operator<<(T value);
    Matrix<T>& operator<<(std::vector<T> values);

    Matrix<T> operator+(Matrix<T> mat2) const;
    Matrix<T> operator+(float number) const;
    Matrix<T> operator-(Matrix<T> mat2) const;

    Matrix<T> operator/(float value) const;
    Matrix<T> operator/(Matrix<T> mat2) const;

    T operator()(int i, int j) const;
    // ===============================[操作符声明结束]=============================

private:
    // ==============================[私有方法]==============================
    T laplaceDeterminant(T **data, int dim);
    // ==============================[私有方法定义结束]==============================

public:
    // ==============================[公有方法]==============================
    Matrix<T> conv2d(Matrix<T>* kernel, int stride, int padding);
    Matrix<T> getKernelSlice(int out_ch, int in_ch);
    Matrix<T> Translate() const;
    Matrix<T> sqrt();
    Matrix<T> rotate180();

    Size shape() const;
    T det();

    T& at(int i, int j);
    T& at(int i, int j, int ch);

    // Const accessors (for read-only access to const Matrix)
    const T& at(int i, int j) const;
    const T& at(int i, int j, int ch) const;

    // Raw data pointer
    T* data_ptr() { return data; }
    const T* data_ptr() const { return data; }

    // Device management
    Device device() const { return device_; }
    void to(Device target);  // Move data between CPU/GPU
    void cuda() { to(Device::GPU); }
    void cpu()  { to(Device::CPU); }

    // Fused optimizer updates (single SIMD pass)
    void fused_adam_update(const Matrix<T>& grad, Matrix<T>& m, Matrix<T>& v,
                           T beta1, T beta2, T m_corr, T v_corr, T lr, T eps);
    void fused_rmsprop_update(const Matrix<T>& grad, Matrix<T>& v,
                               T beta, T lr, T eps);
    void fused_adagrad_update(const Matrix<T>& grad, Matrix<T>& v,
                               T lr, T eps);

    void check();
    T max();

    Matrix<T> getChannel(int channel);
    const Matrix<T>& setChannel(Matrix<T>& mat, int channel);

    void Analysis(const std::string& title = "Matrix");
    // ==============================[公有方法定义结束]==============================
};

#endif // NPCORE_CORE_MATRIX_HPP
