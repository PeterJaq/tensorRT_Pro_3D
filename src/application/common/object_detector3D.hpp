#ifndef OBJECT_DETECTOR3D_HPP
#define OBJECT_DETECTOR3D_HPP

#include <vector>

namespace ObjectDetector3D{

    struct Box3D{
        float x, y, z, x_size, y_size, z_size, ry, confidence;
        int class_label;

        Box3D() = default;

        Box3D(float x, float y, float z, float x_size, float y_size, float z_size, float ry, float confidence, int class_label)
        :x(x), y(y), z(z), x_size(x_size), y_size(y_size), z_size(z_size), ry(ry), confidence(confidence), class_label(class_label){}
    };

    typedef std::vector<Box3D> Box3DArray;
};


#endif // OBJECT_DETECTOR3D_HPP