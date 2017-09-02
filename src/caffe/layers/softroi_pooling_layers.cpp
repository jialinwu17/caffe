// ------------------------------------------------------------------
// R-FCN
// Written by Yi Li
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/softroi_pooling_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
  template <typename Dtype>
  void SOFTROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
  void SOFTROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    N_ = bottom[0]->shape(0);
    //channels_ = bottom[0]->channels();
    //CHECK_EQ(channels_, output_dim_*group_size_*group_size_)
    //  << "input channel number does not match layer parameters";
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    top[0]->Reshape(
      bottom[1]->num(), output_dim_, K /*pooled_height_*/, 1 /*pooled_width_*/);
    mask_spat_sum_.Reshape(
      N_, output_dim_, K /*pooled_height_*/, 1 /*pooled_width_*/);
  }

  template <typename Dtype>
  void SOFTROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_mask = bottom[1]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* mask_spat_sum = mask_spat_sum_.mutable_cpu_data();
    int count = bottom[0]->count();
    caffe_set(top[0]->count(), Dtype(0), top_data);
    caffe_set(N_*K*output_dim_, Dtype(0), mask_spat_sum);
    
    // NOLINT_NEXT_LINE(whitespace/operators)
    
    for (int index = 0; index < count ; index ++){
      int mask_index = index % ( K * height_ * width_ );
      int n = index / output_dim_ / K / height_ / width_ ;
      int k = (index / height_ / width_)% K ;
      int m = (index / height_ / width_ / K)% output_dim_;
      mask_index += n* K * height_ * width_;
      top_data[n*output_dim_*K + m*K + k] += bottom_data[index]*bottom_mask[mask_index];
      mask_spat_sum[n*output_dim_*K + m*K + k] += bottom_mask[mask_index];
    }
    
    //printf("%d,%d,%d\n",N_,output_dim_,K);
    for(int index = 0; index <N_*output_dim_*K ; index ++)
      if (mask_spat_sum[index] > 0)top_data[index] /= mask_spat_sum[index];
       
    
  }

  template <typename Dtype>
  void SOFTROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_mask = bottom[1]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff_data = bottom[0]->mutable_cpu_diff();
    Dtype* bottom_diff_mask= bottom[1]->mutable_cpu_diff();
    const int bottom_count = bottom[0]->count();
    const Dtype* mask_spat_sum = mask_spat_sum_.cpu_data();
    caffe_set(bottom[1]->count(), Dtype(0), bottom_diff_mask);
    caffe_set(bottom_count, Dtype(0), bottom_diff_data);
    const int count = bottom[0]->count();
    for (int index = 0; index < count ; index ++){
      int idx = index % (height_ * width_); 
      int mask_index = index % ( K * height_ * width_ );
      int n = index / output_dim_ / K / height_ / width_ ;
      int k = (index / height_ / width_)% K ;
      int m = (index / height_ / width_ / K)% output_dim_;
      int tmp = n* K * height_ * width_;
      mask_index = mask_index + tmp;
      if(mask_spat_sum[n*output_dim_*K + m*K + k ]> 0){
        bottom_diff_data[index] = top_diff[n*output_dim_*K + m*K + k]*bottom_mask[mask_index]/mask_spat_sum[n*output_dim_*K + m*K + k];
        //printf("%f,%f\n",bottom_diff_data[index],top_diff[n*output_dim*K + m*K + k]);
        int data_start =  (n*output_dim_*K + m*K + k)*height_*width_;
        int mask_start = (n*K + k)*height_*width_;
        bottom_diff_mask[mask_index] = bottom_diff_mask[mask_index]+ top_diff[n*output_dim_*K + m*K + k]*bottom_data[index]/mask_spat_sum[n*output_dim_*K + m*K + k];
        for (int h = 0; h < height_; h ++){
        	for (int w = 0; w < width_ ; w++ ){
        		bottom_diff_mask[mask_start + idx] = bottom_diff_mask[mask_start + idx] -  top_diff[n*output_dim_*K + m*K + k]*bottom_data[data_start+h*width_+w]*bottom_mask[mask_start+h*width_+w]/mask_spat_sum[n*output_dim_*K + m*K + k]/mask_spat_sum[n*output_dim_*K + m*K + k];
        	}
        }
      }
    }
    
  }
//#ifdef CPU_ONLY
//  STUB_GPU(SOFTROIPoolingLayer);
//#endif

  INSTANTIATE_CLASS(SOFTROIPoolingLayer);
  REGISTER_LAYER_CLASS(SOFTROIPooling);

}  // namespace caffe
