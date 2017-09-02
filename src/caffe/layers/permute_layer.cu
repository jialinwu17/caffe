#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/permute_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PermuteForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, Dtype* const top_data, int* map_idx) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % width;
    const int ph = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    int new_idx = n *  width * height * channels + ph * channels *  width + pw * channels + c;
    
    map_idx[index] = new_idx;
    top_data[new_idx] = bottom_data[index];
    }
}
template <typename Dtype>
void PermuteLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  int *map_idx = map_idx_.mutable_gpu_data();
  // We'll output the mask to top[1] if it's of size >1.
  PermuteForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,height_, width_,top_data, map_idx);
  CUDA_POST_KERNEL_CHECK;
  
}


template <typename Dtype>
__global__ void PermuteBackward(const int nthreads, const Dtype* const top_diff,  const int* map_idx, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    bottom_diff[index] = top_diff[map_idx[index]];
  }
}

template <typename Dtype>
void PermuteLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  PermuteBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count, top_diff,map_idx_.gpu_data(), bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PermuteLayer);


}  // namespace caffe
