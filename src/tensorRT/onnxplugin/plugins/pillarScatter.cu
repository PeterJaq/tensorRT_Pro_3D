
#include <onnxplugin/onnxplugin.hpp>
#include <cuda_fp16.hpp>

using namespace ONNXPlugin;

template <typename Element>
__global__ void scatterBEV_kernel(const Element *pillar_features_data,
          const unsigned int *coords_data, const unsigned int *params_data,
          unsigned int featureX, unsigned int featureY,
          Element *spatial_feature_data)
{
    int pillar_idx = blockIdx.x * PILLARS_PER_BLOCK + threadIdx.x;
    int valid_pillars_inBlock = PILLARS_PER_BLOCK;
    const int num_pillars = params_data[0];
    int valid_blocks = (num_pillars+PILLARS_PER_BLOCK-1)/PILLARS_PER_BLOCK;
    if(blockIdx.x >= valid_blocks) return;
    if(blockIdx.x == (valid_blocks-1)) {
      valid_pillars_inBlock = num_pillars % PILLARS_PER_BLOCK;
    }
    valid_pillars_inBlock = (valid_pillars_inBlock==0) ? PILLARS_PER_BLOCK : valid_pillars_inBlock;
    __shared__ Element pillarSM[PILLARS_PER_BLOCK][PILLAR_FEATURE_SIZE]; //pillar*64
    for (int i = 0; i < valid_pillars_inBlock; i++)
    {
      pillarSM[i][threadIdx.x] = pillar_features_data[ (blockIdx.x * PILLARS_PER_BLOCK +i)*PILLAR_FEATURE_SIZE + threadIdx.x];
    }
    __syncthreads();
    if(pillar_idx >= num_pillars) return;
    uint4 coord = ((const uint4 *)coords_data)[pillar_idx];
    unsigned int x = coord.w;
    unsigned int y = coord.z;
    for (int i = 0; i < PILLAR_FEATURE_SIZE; i++)
    {
      spatial_feature_data[i*featureY*featureX + y*featureX + x] = pillarSM[threadIdx.x][i];
    }
}

template <typename Element>
int pillarScatterKernelLaunch(
  int max_pillar_num,
  int num_features,
  const Element *pillar_features_data,
  const unsigned int *coords_data,
  const unsigned int *params_data,
  unsigned int featureX, unsigned int featureY,
  Element *spatial_feature_data,
  cudaStream_t stream)
{
    dim3 blocks( (featureX*featureY+PILLARS_PER_BLOCK-1)/PILLARS_PER_BLOCK);
    dim3 threads(PILLARS_PER_BLOCK);
    scatterBEV_kernel<Element><<<blocks, threads, 0, stream>>>(pillar_features_data, coords_data, params_data, featureX, featureY, spatial_feature_data);
    auto err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

template int pillarScatterKernelLaunch<half>(
  int max_pillar_num,
  int num_features,
  const half *pillar_features_data,
  const unsigned int *coords_data,
  const unsigned int *params_data,
  unsigned int featureX, unsigned int featureY,
  half *spatial_feature_data,
  cudaStream_t stream);

template int pillarScatterKernelLaunch<float>(
  int max_pillar_num,
  int num_features,
  const float *pillar_features_data,
  const unsigned int *coords_data,
  const unsigned int *params_data,
  unsigned int featureX, unsigned int featureY,
  float *spatial_feature_data,
  cudaStream_t stream);


class PillarScatter : public TRTPlugin {
public:
	SetupPlugin(PillarScatter);

	virtual void config_finish() override{
		 
		// INFO("init hswish config: %s", config_->info_.c_str());
		// INFO("weights = %d", config_->weights_.size());
		// for(int i = 0; i < config_->weights_.size(); ++i){
		// 	auto& w = config_->weights_[i];
		// 	if(w->type() == TRT::DataType::Float16){
		// 		INFO("Weight[%d] shape is %s, dtype = %s, value[0] = %f", i, w->shape_string(), data_type_string(w->type()), float(w->at<__half>(0)));
		// 	}else{
		// 		INFO("Weight[%d] shape is %s, dtype = %s, value[0] = %f", i, w->shape_string(), data_type_string(w->type()), w->at<float>(0));
		// 	}
		// }
	}

	virtual std::shared_ptr<LayerConfig> new_config() override{
		auto cfg = TRTPlugin::new_config();

		//cfg->support_dtype_set_ = {nvinfer1::DataType::kHALF, nvinfer1::DataType::kFLOAT};
		cfg->support_dtype_set_ = {nvinfer1::DataType::kFLOAT};
		return cfg;
	}

	nvinfer1::DimsExprs getOutputDimensions(
        	int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) const noexcept{
        assert(outputIndex == 0);
        nvinfer1::DimsExprs output;
        output.nbDims = 4;
        output.d[0] = exprBuilder.constant(1);
        output.d[1] = inputs[0].d[1];
        output.d[2] = exprBuilder.constant(feature_y_size_);
        output.d[3] = exprBuilder.constant(feature_x_size_);
        return output;
	}
    
    nvinfer1::DataType getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept{
        return inputTypes[0];
    }

    bool supportsFormatCombination(
                int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) const noexcept{ 
        assert(nbInputs == 3);
        assert(nbOutputs == 1);
        const PluginTensorDesc& in = inOut[pos];
        if (pos == 0)
        {
            return (in.type == nvinfer1::DataType::kFLOAT || in.type == nvinfer1::DataType::kHALF) && (in.format == TensorFormat::kLINEAR);
        }
        if (pos == 1)
        {
            return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
        }
        if (pos == 2)
        {
            return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
        }
        if (pos == 3)
        {
            return (in.type == inOut[0].type) && (in.format == TensorFormat::kLINEAR || in.format == TensorFormat::kHWC8);
        }
        return false;
    }

    size_t getSerializationSize() const noexcept{
        return 3 * sizeof(size_t);
    }

    void serialize(void* buffer) const noexcept{
        char* d = reinterpret_cast<char*>(buffer);
        writeToBuffer<size_t>(d, feature_y_size_);
        writeToBuffer<size_t>(d, feature_x_size_);
    }


	int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc, onst nvinfer1::PluginTensorDesc* outputDesc, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) override{
    
    try
    {
        int maxPillarNum = inputDesc[0].dims.d[0];
        int numFeatures = inputDesc[0].dims.d[1];
        
        nvinfer1::DataType inputType = inputDesc[0].type;

        auto coords_data = static_cast<const unsigned int *>(inputs[1]);
        auto params_data = static_cast<const unsigned int *>(inputs[2]);

        unsigned int featureY = feature_y_size_;
        unsigned int featureX = feature_x_size_;

        int status = -1;

        if(inputType == nvinfer1::DataType::kHALF){
            auto pillar_features_data = static_cast<const half *>(inputs[0]);
            auto spatial_feature_data = static_cast<half *>(outputs[0]);
            cudaMemsetAsync(spatial_feature_data, 0, numFeatures*featureY*featureX * sizeof(half), stream);
            status = pillarScatterKernelLaunch<half>(
                maxPillarNum,
                numFeatures,
                pillar_features_data,
                coords_data,
                params_data,
                featureX,
                featureY,
                spatial_feature_data,
                stream
                );
            assert(status == STATUS_SUCCESS);
            return status;
        }
        else if(inputType == nvinfer1::DataType::kFLOAT){
            auto pillar_features_data = static_cast<const float *>(inputs[0]);
            auto spatial_feature_data = static_cast<float *>(outputs[0]);
            cudaMemsetAsync(spatial_feature_data, 0, numFeatures*featureY*featureX * sizeof(float), stream);
            status = pillarScatterKernelLaunch<float>(
                maxPillarNum,
                numFeatures,
                pillar_features_data,
                coords_data,
                params_data,
                featureX,
                featureY,
                spatial_feature_data,
                stream
                );
            assert(status == STATUS_SUCCESS);
            return status;
        }
        else{
            assert(status == STATUS_SUCCESS);
            return status;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
    return -1;
    }
};

RegisterPlugin(PillarScatter);