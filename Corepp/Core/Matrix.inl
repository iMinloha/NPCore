// =====================================[Matrix 模板实现]====================================

template<typename T>
Matrix<T> Matrix<T>::operator+(Matrix<T> mat2) const {
    COREPP_ASSERT(row == mat2.row && col == mat2.col, "Matrix shape mismatch");
    Matrix<T> res(row, col);
    int total = row * col;
#ifdef __AVX__
    int i = 0;
    for (; i + 8 <= total; i += 8) {
        __m256 a = _mm256_loadu_ps(&data[i]);
        __m256 b = _mm256_loadu_ps(&mat2.data[i]);
        _mm256_storeu_ps(&res.data[i], _mm256_add_ps(a, b));
    }
    for (; i < total; ++i) res.data[i] = data[i] + mat2.data[i];
#elif defined(__ARM_NEON)
    int i = 0;
    for (; i + 4 <= total; i += 4) {
        float32x4_t a = vld1q_f32(&data[i]);
        float32x4_t b = vld1q_f32(&mat2.data[i]);
        vst1q_f32(&res.data[i], vaddq_f32(a, b));
    }
    for (; i < total; ++i) res.data[i] = data[i] + mat2.data[i];
#else
    for (int i = 0; i < total; ++i) res.data[i] = data[i] + mat2.data[i];
#endif
    return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(float number) const {
    Matrix<T> res(row, col);
    int total = row * col;
#ifdef __AVX__
    __m256 num = _mm256_set1_ps(number);
    int i = 0;
    for (; i + 8 <= total; i += 8) {
        __m256 a = _mm256_loadu_ps(&data[i]);
        _mm256_storeu_ps(&res.data[i], _mm256_add_ps(a, num));
    }
    for (; i < total; ++i) res.data[i] = data[i] + number;
#elif defined(__ARM_NEON)
    float32x4_t num = vmovq_n_f32(number);
    int i = 0;
    for (; i + 4 <= total; i += 4) {
        float32x4_t a = vld1q_f32(&data[i]);
        vst1q_f32(&res.data[i], vaddq_f32(a, num));
    }
    for (; i < total; ++i) res.data[i] = data[i] + number;
#else
    for (int i = 0; i < total; ++i) res.data[i] = data[i] + number;
#endif
    return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(float value) const {
    Matrix<T> res(row, col);
    int total = row * col;
#ifdef __AVX__
    __m256 denom = _mm256_set1_ps(value);
    int i = 0;
    for (; i + 8 <= total; i += 8) {
        __m256 a = _mm256_loadu_ps(&data[i]);
        _mm256_storeu_ps(&res.data[i], _mm256_div_ps(a, denom));
    }
    for (; i < total; ++i) res.data[i] = data[i] / value;
#elif defined(__ARM_NEON)
    float32x4_t denom = vmovq_n_f32(value);
    int i = 0;
    for (; i + 4 <= total; i += 4) {
        float32x4_t a = vld1q_f32(&data[i]);
        vst1q_f32(&res.data[i], vdivq_f32(a, denom));
    }
    for (; i < total; ++i) res.data[i] = data[i] / value;
#else
    for (int i = 0; i < total; ++i) res.data[i] = data[i] / value;
#endif
    return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(Matrix<T> mat2) const {
    COREPP_ASSERT(this->shape() == mat2.shape(), "Matrix shape mismatch");
    Matrix<T> res(row, col);
    int total = row * col;
#ifdef __AVX__
    int i = 0;
    for (; i + 8 <= total; i += 8) {
        __m256 a = _mm256_loadu_ps(&data[i]);
        __m256 b = _mm256_loadu_ps(&mat2.data[i]);
        _mm256_storeu_ps(&res.data[i], _mm256_div_ps(a, b));
    }
    for (; i < total; ++i) res.data[i] = data[i] / mat2.data[i];
#elif defined(__ARM_NEON)
    int i = 0;
    for (; i + 4 <= total; i += 4) {
        float32x4_t a = vld1q_f32(&data[i]);
        float32x4_t b = vld1q_f32(&mat2.data[i]);
        vst1q_f32(&res.data[i], vdivq_f32(a, b));
    }
    for (; i < total; ++i) res.data[i] = data[i] / mat2.data[i];
#else
    for (int i = 0; i < total; ++i) res.data[i] = data[i] / mat2.data[i];
#endif
    return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(Matrix<T> mat2) const {
    COREPP_ASSERT(col == mat2.row, "Matrix multiplication dimension mismatch");
    Matrix<T> result(row, mat2.col);

    // --- GPU-resident path: both operands on GPU, result stays on GPU ---
    if constexpr (std::is_same_v<T, float>) {
        if (device_ == Device::GPU && mat2.device_ == Device::GPU) {
            result.device_ = Device::GPU;
            result.data = (float*)CoreNNSpace::cuda_malloc_device(row * result.col * sizeof(float));
            result.owns_data = true;
            if (CoreNNSpace::cuda_gemm_device(row, mat2.col, col, data, mat2.data, result.data))
                return result;
        }
        // --- CPU fallback: one-shot H2D→kernel→D2H ---
        int total = row * result.col;
        for (int i = 0; i < total; ++i) result.data[i] = T(0);
        if (CoreNNSpace::cuda_gemm_dispatch(row, mat2.col, col, data, mat2.data, result.data))
            return result;
    }

#ifdef __AVX__
    int total = result.row * result.col;
    for (int i = 0; i < total; ++i) result.data[i] = T(0);

    gemm(row, mat2.col, col,
         data, col,
         mat2.data, mat2.col,
         result.data, result.col);
#else
    // Scalar fallback with basic blocking + OpenMP
    constexpr int BLOCK_SIZE = 64;
#pragma omp parallel for collapse(2) if (row * mat2.col > 4096)
    for (int i = 0; i < row; i += BLOCK_SIZE) {
        for (int j = 0; j < mat2.col; j += BLOCK_SIZE) {
            for (int k = 0; k < col; k += BLOCK_SIZE) {
                int i_end = std::min(i + BLOCK_SIZE, row);
                int j_end = std::min(j + BLOCK_SIZE, mat2.col);
                int k_end = std::min(k + BLOCK_SIZE, col);
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        T sum = result.at(ii, jj);
                        for (int kk = k; kk < k_end; ++kk)
                            sum += at(ii, kk) * mat2.at(kk, jj);
                        result.at(ii, jj) = sum;
                    }
                }
            }
        }
    }
#endif

    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(T value) const {
    Matrix<T> result(row, col);
    int total = row * col;
#ifdef __AVX__
    __m256 scalar = _mm256_set1_ps(value);
    int i = 0;
    for (; i + 8 <= total; i += 8) {
        __m256 a = _mm256_loadu_ps(&data[i]);
        _mm256_storeu_ps(&result.data[i], _mm256_mul_ps(a, scalar));
    }
    for (; i < total; ++i) result.data[i] = data[i] * value;
#elif defined(__ARM_NEON)
    float32x4_t scalar = vmovq_n_f32(value);
    int i = 0;
    for (; i + 4 <= total; i += 4) {
        float32x4_t a = vld1q_f32(&data[i]);
        vst1q_f32(&result.data[i], vmulq_f32(a, scalar));
    }
    for (; i < total; ++i) result.data[i] = data[i] * value;
#else
    for (int i = 0; i < total; ++i) result.data[i] = data[i] * value;
#endif
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::getKernelSlice(int out_ch, int in_ch)  {
    int kernel_ch = in_ch + out_ch * channel;
    Matrix<T> slice(row, col);
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < col; ++j)
            slice.at(i, j) = this->at(i, j, kernel_ch);

    return slice;
}

template<typename T>
Matrix<T> Matrix<T>::conv2d(Matrix<T>* kernel, int stride, int padding) {
    COREPP_ASSERT(kernel != nullptr, "Kernel pointer cannot be null");

    int in_channels = channel;
    int out_channels = kernel->channel / in_channels;

    COREPP_ASSERT(in_channels * out_channels == kernel->channel,
                 "Kernel channel mismatch with in/out channels");
    COREPP_ASSERT(stride > 0, "Stride must be positive");

    int k_row = kernel->row;
    int k_col = kernel->col;

    int out_row = (row - k_row + 2 * padding) / stride + 1;
    int out_col = (col - k_col + 2 * padding) / stride + 1;

    COREPP_ASSERT(out_row > 0 && out_col > 0, "Output dimensions must be positive");

    Matrix<T> output(out_row, out_col, out_channels);

    #pragma omp parallel for collapse(3) if (out_row * out_col * out_channels > 1000)
    for (int out_ch = 0; out_ch < out_channels; ++out_ch) {
        for (int i = 0; i < out_row; ++i) {
            for (int j = 0; j < out_col; ++j) {
                T sum = T(0);
                int base_i = i * stride - padding;
                int base_j = j * stride - padding;

                for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                    int kernel_offset = (out_ch * in_channels + in_ch) * k_row * k_col;

                    for (int ki = 0; ki < k_row; ++ki) {
                        int src_i = base_i + ki;
                        if (src_i < 0 || src_i >= row) continue;

                        for (int kj = 0; kj < k_col; ++kj) {
                            int src_j = base_j + kj;
                            if (src_j < 0 || src_j >= col) continue;

                            T pixel_val = this->at(src_i, src_j, in_ch);
                            T kernel_val = kernel->data[kernel_offset + ki * k_col + kj];
                            sum += pixel_val * kernel_val;
                        }
                    }
                }
                output.at(i, j, out_ch) = sum;
            }
        }
    }

    return output;
}

template<typename T>
Matrix<T> Matrix<T>::Translate() const {
    Matrix<T> result(col, row);
    #pragma omp parallel for collapse(2) if (row > 64)
    for (int i = 0; i < row; ++i) for (int j = 0; j < col; ++j) result.at(j, i) = at(i, j);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::sqrt() {
    Matrix<T> result(row, col);
    int total = row * col;
#ifdef __AVX__
    int i = 0;
    for (; i + 8 <= total; i += 8) {
        __m256 a = _mm256_loadu_ps(&data[i]);
        _mm256_storeu_ps(&result.data[i], _mm256_sqrt_ps(a));
    }
    for (; i < total; ++i) result.data[i] = std::sqrt(data[i]);
#elif defined(__ARM_NEON)
    int i = 0;
    for (; i + 4 <= total; i += 4) {
        float32x4_t a = vld1q_f32(&data[i]);
        // NEON doesn't have native sqrt; fall back to scalar
        float tmp[4];
        vst1q_f32(tmp, a);
        for (int k = 0; k < 4; ++k) tmp[k] = std::sqrt(tmp[k]);
        vst1q_f32(&result.data[i], vld1q_f32(tmp));
    }
    for (; i < total; ++i) result.data[i] = std::sqrt(data[i]);
#else
    for (int i = 0; i < total; ++i) result.data[i] = std::sqrt(data[i]);
#endif
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::rotate180() {
    Matrix<T> result(row, col, channel);
    for (int ch = 0; ch < channel; ++ch)
        for (int i = 0; i < row; ++i)
            for (int j = 0; j < col; ++j)
                result.at(row - 1 - i, col - 1 - j, ch) = at(i, j, ch);

    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(Matrix<T> mat2) const {
    COREPP_ASSERT(row == mat2.row && col == mat2.col, "Matrix shape mismatch");
    Matrix<T> res(row, col);
    int total = row * col;
#ifdef __AVX__
    int i = 0;
    for (; i + 8 <= total; i += 8) {
        __m256 a = _mm256_loadu_ps(&data[i]);
        __m256 b = _mm256_loadu_ps(&mat2.data[i]);
        _mm256_storeu_ps(&res.data[i], _mm256_sub_ps(a, b));
    }
    for (; i < total; ++i) res.data[i] = data[i] - mat2.data[i];
#elif defined(__ARM_NEON)
    int i = 0;
    for (; i + 4 <= total; i += 4) {
        float32x4_t a = vld1q_f32(&data[i]);
        float32x4_t b = vld1q_f32(&mat2.data[i]);
        vst1q_f32(&res.data[i], vsubq_f32(a, b));
    }
    for (; i < total; ++i) res.data[i] = data[i] - mat2.data[i];
#else
    for (int i = 0; i < total; ++i) res.data[i] = data[i] - mat2.data[i];
#endif
    return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator&(Matrix<T> mat2) const {
    COREPP_ASSERT(row == mat2.row && col == mat2.col && channel == mat2.channel, "Matrix shape mismatch");
    Matrix<T> res(row, col, channel);
    int total = row * col * channel;
#ifdef __AVX__
    int i = 0;
    for (; i + 8 <= total; i += 8) {
        __m256 a = _mm256_loadu_ps(&data[i]);
        __m256 b = _mm256_loadu_ps(&mat2.data[i]);
        _mm256_storeu_ps(&res.data[i], _mm256_mul_ps(a, b));
    }
    for (; i < total; ++i) res.data[i] = data[i] * mat2.data[i];
#elif defined(__ARM_NEON)
    int i = 0;
    for (; i + 4 <= total; i += 4) {
        float32x4_t a = vld1q_f32(&data[i]);
        float32x4_t b = vld1q_f32(&mat2.data[i]);
        vst1q_f32(&res.data[i], vmulq_f32(a, b));
    }
    for (; i < total; ++i) res.data[i] = data[i] * mat2.data[i];
#else
    for (int i = 0; i < total; ++i) res.data[i] = data[i] * mat2.data[i];
#endif
    return res;
}

template<typename T>
T Matrix<T>::det() {
    COREPP_ASSERT(row == col, "Det can only be calculated for square matrix");

    T** temp = new T*[row];
    for (int i = 0; i < row; ++i) {
        temp[i] = new T[col];
        for (int j = 0; j < col; ++j)
            temp[i][j] = at(i, j);
    }

    T result = laplaceDeterminant(temp, row);

    for (int i = 0; i < row; ++i) delete[] temp[i];
    delete[] temp;

    return result;
}

template<typename T>
T Matrix<T>::laplaceDeterminant(T **data, int dim) {
    if (dim == 1) return data[0][0];
    else if (dim == 2) return data[0][0] * data[1][1] - data[0][1] * data[1][0];
    else {
        T result = 0;
        for (int i = 0; i < dim; i++) {
            T **temp = new T *[dim - 1];
            for (int j = 0; j < dim - 1; j++) temp[j] = new T[dim - 1];
            for (int j = 1; j < dim; j++) {
                int count = 0;
                for (int k = 0; k < dim; k++) {
                    if (k == i) continue;
                    temp[j - 1][count] = data[j][k];
                    count++;
                }
            }
            if (i % 2 == 0) result += data[0][i] * laplaceDeterminant(temp, dim - 1);
            else result -= data[0][i] * laplaceDeterminant(temp, dim - 1);

            for (int j = 0; j < dim - 1; j++) delete[] temp[j];
            delete[] temp;
        }
        return result;
    }
}

template<typename T>
Size Matrix<T>::shape() {
    return {row, col, channel};
}

template<typename T>
T& Matrix<T>::at(int i, int j) {
    return data[i * col + j];
}

template<typename T>
const T& Matrix<T>::at(int i, int j) const {
    return data[i * col + j];
}

template<typename T>
T& Matrix<T>::at(int i, int j, int ch) {
    return data[i * col * channel + j * channel + ch];
}

template<typename T>
const T& Matrix<T>::at(int i, int j, int ch) const {
    return data[i * col * channel + j * channel + ch];
}

template<typename T>
Matrix<T> &Matrix<T>::operator<<(T value) {
    if (currentIndex < row * col) {
        size_t r = currentIndex / col;
        size_t c = currentIndex % col;
        data[r * col + c] = value;
        currentIndex++;
    } else COREPP_ASSERT(false, "Append out of range");
    return *this;
}

template<typename T>
Matrix<T> &Matrix<T>::operator<<(std::vector<T> values) {
    for (auto &value : values) {
        if (currentIndex < row * col) {
            size_t r = currentIndex / col;
            size_t c = currentIndex % col;
            data[r * col + c] = value;
            currentIndex++;
        } else COREPP_ASSERT(false, "Append out of range");
    }
    return *this;
}

template<typename T>
T Matrix<T>::operator()(int i, int j) const {
    return data[i * col + j];
}

template<typename T>
void Matrix<T>::check() {
    if (this->row <=0 || this->col <=0)
        throw std::runtime_error("Invalid gradient matrix dimensions");
    for(int r=0; r < this->row; ++r)
        for(int c=0; c < this->col; ++c)
            if (std::isnan(this->at(r,c)))
                throw std::runtime_error("NaN detected in gradients");
}

template<typename T>
T Matrix<T>::max() {
    T max = 0;
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++) if (at(i, j) > max) max = at(i, j);

    return max;
}

template<typename T>
Matrix<T> Matrix<T>::getChannel(int channel) {
    Matrix<T> result(row, col);
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            result.at(i, j) = at(i, j, channel);
    return result;
}

template<typename T>
const Matrix<T>& Matrix<T>::setChannel(Matrix<T>& mat, int channel) {
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            at(i, j, channel) = mat.at(i, j);
    return *this;
}

template<typename T>
void Matrix<T>::Analysis(const std::string& title) {
    // --- Smart number formatting: auto scientific for tiny/huge, fixed for normal ---
    auto fmt = [](T v) -> std::string {
        T av = std::abs(v);
        std::ostringstream oss;
        if (av == T(0)) {
            return "0";
        } else if (av < T(1e-6) || av >= T(1e7)) {
            oss << std::scientific << std::setprecision(4) << v;
        } else if (av < T(0.001)) {
            oss << std::scientific << std::setprecision(4) << v;
        } else if (av < T(1)) {
            oss << std::fixed << std::setprecision(6) << v;
        } else if (av < T(10)) {
            oss << std::fixed << std::setprecision(4) << v;
        } else if (av < T(100)) {
            oss << std::fixed << std::setprecision(3) << v;
        } else if (av < T(1000)) {
            oss << std::fixed << std::setprecision(1) << v;
        } else {
            oss << std::fixed << std::setprecision(0) << v;
        }
        return oss.str();
    };

    // --- Sign prefix: + / - / ~ (near-zero) ---
    auto sign = [](T v) -> char {
        if (v > T(1e-9))  return '+';
        if (v < T(-1e-9)) return '-';
        return '~';
    };

    // --- Compute column width ---
    int cellW = 8;  // minimum
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < col; ++j)
            for (int c = 0; c < (channel > 1 ? channel : 1); ++c) {
                T v = (channel == 1) ? at(i, j) : at(i, j, c);
                int len = (int)fmt(std::abs(v)).length();
                if (len + 1 > cellW) cellW = len + 1;
            }

    int totalW = col * (cellW + 1) + 3;
    int lblW   = (row > 9) ? 3 : 2;

    auto hline = [&](char l, char /*m*/, char r, char fill) {
        std::cout << std::string(lblW, ' ') << l << std::string(totalW - 2, fill) << r << '\n';
    };

    // --- Header ---
    hline('+', '=', '+', '=');
    std::cout << std::string(lblW, ' ') << "| " << std::left << std::setw(totalW - 4)
              << title << " [" << row << "x" << col;
    if (channel > 1) std::cout << "x" << channel;
    std::cout << "]\n";
    hline('+', '=', '+', '=');

    // --- Data ---
    int slices = (channel > 1) ? channel : 1;
    for (int ch = 0; ch < slices; ++ch) {
        if (channel > 1 && ch > 0) {
            std::cout << std::string(lblW, ' ') << "| -- ch:" << ch
                      << std::string(totalW - 12, ' ') << "|\n";
        }
        for (int i = 0; i < row; ++i) {
            std::cout << std::setw(lblW - 1) << i << " |";
            for (int j = 0; j < col; ++j) {
                T v = (channel == 1) ? at(i, j) : at(i, j, ch);
                std::cout << sign(v) << std::setw(cellW) << std::right << fmt(std::abs(v));
            }
            std::cout << " |\n";
        }
    }
    hline('+', '-', '+', '-');
}

// ====================================[Device Transfer]=================================

template<typename T>
void Matrix<T>::to(Device target) {
    if (device_ == target) return;
    int n = row * col * channel;
    (void)n;

    if (target == Device::GPU) {
#ifdef COREPP_ENABLE_CUDA
        T* gpu_data = (T*)cuda_corepp_malloc(n * sizeof(T));
        cuda_corepp_memcpy_h2d(gpu_data, data, n * sizeof(T));
        if (owns_data) delete[] data;
        data = gpu_data;
        device_ = Device::GPU;
        owns_data = true;
#endif
    } else {
#ifdef COREPP_ENABLE_CUDA
        T* cpu_data = new T[n];
        cuda_corepp_memcpy_d2h(cpu_data, data, n * sizeof(T));
        if (owns_data) cuda_corepp_free(data);
        data = cpu_data;
        device_ = Device::CPU;
        owns_data = true;
#endif
    }
}

// ====================================[Fused Optimizer Updates]=================================

template<typename T>
void Matrix<T>::fused_adam_update(const Matrix<T>& grad, Matrix<T>& m, Matrix<T>& v,
                                   T beta1, T beta2, T m_corr, T v_corr, T lr, T eps) {
    int total = row * col;
    T b1 = beta1, b2 = beta2;
    T one_m_b1 = T(1) - b1, one_m_b2 = T(1) - b2;

#ifdef __AVX__
    __m256 vb1 = _mm256_set1_ps(b1), vb2 = _mm256_set1_ps(b2);
    __m256 v1mb1 = _mm256_set1_ps(one_m_b1), v1mb2 = _mm256_set1_ps(one_m_b2);
    __m256 vmc = _mm256_set1_ps(m_corr), vvc = _mm256_set1_ps(v_corr);
    __m256 vlr = _mm256_set1_ps(lr), veps = _mm256_set1_ps(eps);

    int i = 0;
    for (; i + 8 <= total; i += 8) {
        __m256 g = _mm256_loadu_ps(&grad.data[i]);
        __m256 mm = _mm256_loadu_ps(&m.data[i]);
        __m256 vv = _mm256_loadu_ps(&v.data[i]);

        // m = beta1 * m + (1-beta1) * g
        mm = _mm256_add_ps(_mm256_mul_ps(vb1, mm), _mm256_mul_ps(v1mb1, g));
        // v = beta2 * v + (1-beta2) * g^2
        vv = _mm256_add_ps(_mm256_mul_ps(vb2, vv), _mm256_mul_ps(v1mb2, _mm256_mul_ps(g, g)));

        _mm256_storeu_ps(&m.data[i], mm);
        _mm256_storeu_ps(&v.data[i], vv);

        // w -= lr * (m_hat) / (sqrt(v_hat) + eps)
        __m256 mhat = _mm256_mul_ps(vmc, mm);
        __m256 vhat = _mm256_mul_ps(vvc, vv);
        __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(vhat), veps);
        __m256 update = _mm256_div_ps(_mm256_mul_ps(vlr, mhat), denom);

        __m256 w = _mm256_loadu_ps(&data[i]);
        _mm256_storeu_ps(&data[i], _mm256_sub_ps(w, update));
    }
    for (; i < total; ++i) {
        T g = grad.data[i];
        m.data[i] = b1 * m.data[i] + one_m_b1 * g;
        v.data[i] = b2 * v.data[i] + one_m_b2 * g * g;
        T mhat = m_corr * m.data[i];
        T vhat = v_corr * v.data[i];
        data[i] -= lr * mhat / (std::sqrt(vhat) + eps);
    }
#else
    for (int i = 0; i < total; ++i) {
        T g = grad.data[i];
        m.data[i] = b1 * m.data[i] + one_m_b1 * g;
        v.data[i] = b2 * v.data[i] + one_m_b2 * g * g;
        T mhat = m_corr * m.data[i];
        T vhat = v_corr * v.data[i];
        data[i] -= lr * mhat / (std::sqrt(vhat) + eps);
    }
#endif
}

template<typename T>
void Matrix<T>::fused_rmsprop_update(const Matrix<T>& grad, Matrix<T>& v,
                                      T beta, T lr, T eps) {
    int total = row * col;
    T b = beta, one_m_b = T(1) - b;

#ifdef __AVX__
    __m256 vb = _mm256_set1_ps(b), v1mb = _mm256_set1_ps(one_m_b);
    __m256 vlr = _mm256_set1_ps(lr), veps = _mm256_set1_ps(eps);

    int i = 0;
    for (; i + 8 <= total; i += 8) {
        __m256 g = _mm256_loadu_ps(&grad.data[i]);
        __m256 vv = _mm256_loadu_ps(&v.data[i]);

        vv = _mm256_add_ps(_mm256_mul_ps(vb, vv), _mm256_mul_ps(v1mb, _mm256_mul_ps(g, g)));
        _mm256_storeu_ps(&v.data[i], vv);

        __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(vv), veps);
        __m256 update = _mm256_div_ps(_mm256_mul_ps(vlr, g), denom);

        __m256 w = _mm256_loadu_ps(&data[i]);
        _mm256_storeu_ps(&data[i], _mm256_sub_ps(w, update));
    }
    for (; i < total; ++i) {
        T g = grad.data[i];
        v.data[i] = b * v.data[i] + one_m_b * g * g;
        data[i] -= lr * g / (std::sqrt(v.data[i]) + eps);
    }
#else
    for (int i = 0; i < total; ++i) {
        T g = grad.data[i];
        v.data[i] = b * v.data[i] + one_m_b * g * g;
        data[i] -= lr * g / (std::sqrt(v.data[i]) + eps);
    }
#endif
}

template<typename T>
void Matrix<T>::fused_adagrad_update(const Matrix<T>& grad, Matrix<T>& v,
                                      T lr, T eps) {
    int total = row * col;

#ifdef __AVX__
    __m256 vlr = _mm256_set1_ps(lr), veps = _mm256_set1_ps(eps);

    int i = 0;
    for (; i + 8 <= total; i += 8) {
        __m256 g = _mm256_loadu_ps(&grad.data[i]);
        __m256 vv = _mm256_loadu_ps(&v.data[i]);

        vv = _mm256_add_ps(vv, _mm256_mul_ps(g, g));
        _mm256_storeu_ps(&v.data[i], vv);

        __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(vv), veps);
        __m256 update = _mm256_div_ps(_mm256_mul_ps(vlr, g), denom);

        __m256 w = _mm256_loadu_ps(&data[i]);
        _mm256_storeu_ps(&data[i], _mm256_sub_ps(w, update));
    }
    for (; i < total; ++i) {
        T g = grad.data[i];
        v.data[i] += g * g;
        data[i] -= lr * g / (std::sqrt(v.data[i]) + eps);
    }
#else
    for (int i = 0; i < total; ++i) {
        T g = grad.data[i];
        v.data[i] += g * g;
        data[i] -= lr * g / (std::sqrt(v.data[i]) + eps);
    }
#endif
}

// ====================================[模板实现结束]=================================
