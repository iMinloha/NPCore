#ifndef NPCORE_CORE_SIZE_H
#define NPCORE_CORE_SIZE_H

// =================================[Size 尺寸类]================================
// 仅作参数类型传递使用

class Size {
public:
    int width;
    int height;
    int channel;

    Size(int width, int height, int channel);

    // 尺寸判断
    bool operator==(Size size) const;

    // 尺寸拷贝
    Size& operator=(Size size);
};

#endif // NPCORE_CORE_SIZE_H
