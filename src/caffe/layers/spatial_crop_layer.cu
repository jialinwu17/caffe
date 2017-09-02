#include <vector>

#include "caffe/layers/spatial_crop_layer.hpp"

namespace caffe {

// Copy (one line per thread) from one array to another, with arbitrary
// strides in the last two dimensions.
template <typename Dtype>
__global__ void spatial_crop_forward(const int n, const int height, const int width, const int channels, const int copy_h, const int copy_w, const int num_crops,
    const Dtype* bottom_data, Dtype* top_data, int* index_map) {
  CUDA_KERNEL_LOOP(index, n) {
     int pw = index % copy_w;
     int ph = (index / copy_w) % copy_h;
     int crops_h_index = (index / copy_w / copy_h/channels) % (num_crops);
     int crops_w_index = (index / copy_w / copy_h/channels/num_crops) % (num_crops);
     int bottom_c = (index / copy_w / copy_h) %channels;
     int bottom_h = crops_h_index*copy_h + ph;
     int bottom_w = crops_w_index*copy_w + pw;
     int n = (index / copy_w / copy_h/channels/num_crops/num_crops) ;
     int bottom_index = ((n* channels + bottom_c)*height+ bottom_h)*width +bottom_w;
     top_data[index] = bottom_data[bottom_index];
     index_map[bottom_index] = index;
  }
}
template <typename Dtype>
__global__ void spatial_crop_backward(const int n,const int* index_map,  const Dtype* top_diff, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, n) {
     bottom_diff[index] = top_diff[index_map[index]];
  }
}
template <typename Dtype>
void SpatialCropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* index_map = index_map_.mutable_gpu_data();
  int count = top[0]->count();
  spatial_crop_forward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(count, height_,width_,channels_, copy_h, copy_w,num_crops,bottom_data, top_data,index_map);
}

template <typename Dtype>
void SpatialCropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int count = bottom[0]->count();
  const int* index_map = index_map_.gpu_data();
  spatial_crop_backward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(count, index_map ,top_diff, bottom_diff);

}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialCropLayer);

}  // namespace caffe
