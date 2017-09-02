// ------------------------------------------------------------------
// R-FCN
// Written by Yi Li
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/attention_pooling_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
  template <typename Dtype>
  void AttentionPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    SOFTROIPoolingParameter softroi_pooling_param =
      this->layer_param_.softroi_pooling_param();
    //spatial_scale_ = softroi_pooling_param.spatial_scale();
    //LOG(INFO) << "Spatial scale: " << spatial_scale_;

    CHECK_GT(softroi_pooling_param.output_dim(), 0)
      << "output_dim must be > 0";
    CHECK_GT(softroi_pooling_param.mask_num(), 0)
      << "group_size must be > 0";

    output_dim_ = softroi_pooling_param.output_dim();
    K = softroi_pooling_param.mask_num();
    //pooled_height_ = K;
    //pooled_width_ = K;
  }

  template <typename Dtype>
  void AttentionPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    N_ = bottom[0]->shape(0);
    //channels_ = bottom[0]->channels();
    //CHECK_EQ(channels_, output_dim_*group_size_*group_size_)
    //  << "input channel number does not match layer parameters";
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    top[0]->Reshape(
      bottom[1]->num(), output_dim_, K /*pooled_height_*/, 1 /*pooled_width_*/);
  }

  template <typename Dtype>
  void AttentionPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void AttentionPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
//#ifdef CPU_ONLY
//  STUB_GPU(SOFTROIPoolingLayer);
//#endif

  INSTANTIATE_CLASS(AttentionPoolingLayer);
  REGISTER_LAYER_CLASS(AttentionPooling);

}  // namespace caffe
