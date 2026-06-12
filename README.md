# CorePP — C++ Deep Learning Library

纯 C++20 深度学习库，支持 SIMD (AVX2/NEON) 加速和 CUDA GPU 推理。

## 特性

- **矩阵运算**: GotoBLAS 风格 packed GEMM 微内核，AVX2 SIMD 加速
- **卷积**: im2col+GEMM，Winograd F(2,3) 变换支持
- **循环网络**: RNN (BPTT)、LSTM (4门合并 GEMM)
- **优化器**: SGD、Momentum、Adam、RMSProp、Adagrad
- **CUDA**: 可选 GPU 加速，纯 C 接口兼容 MinGW
- **自动梯度检验**: 数值梯度 vs 分析梯度对比

## 快速开始

```cpp
#include "CorePP.h"
#include "Model.h"
using namespace CoreNNSpace;

int main() {
    // 一行创建全连接网络: 4→8→16→8→4
    auto fnn = nn::FNN({4, 8, 16, 8, 4}, nn::Sigmoid);

    Matrix<float> x(4, 1); x << 3 << 4 << 2 << 1;
    Matrix<float> y(4, 1); y << 1 << 0 << 0 << 0;

    // 训练
    nn::Trainer(fnn, nn::MSE, Optim(fnn.getParams(), Adam, 0.01f))
        .fit(x, y, 300, [](int e, float loss) {
            printf("epoch %d: %.6f\n", e, loss);
        });

    // 推理
    auto out = fnn.forward(x);
    out.Analysis("Prediction");
}
```

## 构建

### 静态库 (默认)
```bash
cmake -G "MinGW Makefiles" -B _build
cmake --build _build
./_build/bin/CorePPDemo
```

### 动态链接库 (DLL)
```bash
# 构建 DLL
cmake -G "MinGW Makefiles" -B _build_shared -DBUILD_SHARED_LIBS=ON
cmake --build _build_shared

# 编译示例 (链接 DLL)
g++ -std=c++20 -I CorePP examples/basic/main.cpp \
    -L_build_shared/lib -lCorePP -fopenmp \
    -o example.exe

# 运行 (确保 DLL 在 PATH)
PATH="$PWD/_build_shared/bin:$PATH" ./example.exe
```

### CUDA 版本 (需要 nvcc + MSVC)
```bash
cmake -G "MinGW Makefiles" -B _build -DCOREPP_ENABLE_CUDA=ON
cmake --build _build
```

## 文档

| 文档 | 内容 |
|------|------|
| [快速入门](docs/QuickStart.md) | 安装、构建、第一个模型 |
| [API 参考](docs/API.md) | 所有类和函数 |
| [层说明](docs/Layers.md) | Linear、Conv2d、RNN、LSTM |
| [CUDA 指南](docs/CUDA.md) | GPU 加速配置 |

## 项目结构

```
CorePP/
├── Core/           # 矩阵、GEMM、随机数
├── Layers/         # 神经网络层
├── Optimizers/     # 优化算法
├── Cuda/           # CUDA 后端
├── Utils/          # 工具函数
├── tests/          # 单元测试
├── examples/       # 使用示例
├── docs/           # 文档
├── cmake/          # CMake 模块
├── CorePP.h        # 主头文件
└── Model.h         # 高层 API
```

## License

MIT
