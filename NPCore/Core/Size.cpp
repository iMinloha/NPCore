#include "Core/Size.h"

Size::Size(int width, int height, int channel)
    : width(width), height(height), channel(channel) {}

bool Size::operator==(Size size) const {
    return width == size.width && height == size.height && channel == size.channel;
}

Size& Size::operator=(Size size) {
    width = size.width;
    height = size.height;
    channel = size.channel;
    return *this;
}
