#ifndef COREPP_CORE_SIZE_H
#define COREPP_CORE_SIZE_H

// =================================[Size 尺寸类]================================
// 仅作参数类型传递使用

class Size {
public:
    int width;
    int height;
    int channel;

    Size(int width, int height, int channel)
        : width(width), height(height), channel(channel) {}

    // 尺寸判断
    bool operator==(Size size) const {
        return width == size.width && height == size.height && channel == size.channel;
    }

    // 尺寸拷贝
    Size& operator=(Size size) {
        width = size.width;
        height = size.height;
        channel = size.channel;
        return *this;
    }
};

#endif // COREPP_CORE_SIZE_H
