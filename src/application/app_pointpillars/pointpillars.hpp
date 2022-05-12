#ifndef YOLO_HPP
#define YOLO_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include <common/trt_tensor.hpp>
#include <common/object_detector3D.hpp>
 
template<typename PointT>

namespace PointPillars{

    using namespace std;
    using namespace ObjectDetector;

    enum class NMSMethod : int{
        CPU = 0,         // General, for estimate mAP
        FastGPU = 1      // Fast NMS with a small loss of accuracy in corner cases
    };
    
    void pcl_to_tensor(const pcl::PointCloud<PointT> pointcloud, shared_ptr<TRT::Tensor>& tensor, Type type, int ibatch);

    class Infer{
    public:
        virtual shared_future<Box3DArray> commit(const pcl::PointCloud<PointT> pointcloud) = 0;
        virtual vector<shared_future<Box3DArray>> commits(const vector<pcl::PointCloud<PointT>>& pointclouds) = 0;
    };

    shared_ptr<Infer> create_infer(
        const string& engine_file, int gpuid,
        float confidence_threshold=0.25f, float nms_threshold=0.5f,
        NMSMethod nms_method = NMSMethod::FastGPU, int max_objects = 1024,
        bool use_multi_preprocess_stream = false
    );
    const char* type_name(Type type);

}; // namespace Yolo

#endif // YOLO_HPP