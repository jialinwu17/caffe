// --------------------------------------------------------
// R-FCN
// Written by Yi Li, 2016.
// --------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/residual_roi_pooling_layer.hpp"
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {

  template <typename Dtype>
  __global__ void ResidualROIPoolingForward(
    const int nthreads,
    const Dtype* bottom_data,
    const int N,
    const int height, const int width,
    const Dtype* bottom_mask,
    const int output_dim,
    const int K,
    Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (N, output_dim, K)
      for(int h = 0; h < height ; h++ ){
      	for(int w = 0; w < width ; w++){
            int k = index % ( K );
            int m = (index / K)% output_dim;
      		int n = index / output_dim / K ;
      		int mask_index = n* K * height * width + k*height * width + h*width +w;
      		int data_index = n* K*output_dim * height * width + (m*K + k)*height * width + h*width +w;
      		top_data[index] += bottom_data[data_index]*(1.0 + bottom_mask[mask_index]);
      	}
      }
      top_data[index] = top_data[index] /height / width ;
    }
  }

  template <typename Dtype>
  void ResidualROIPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_mask = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    
    int count = top[0]->count();
    caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
    
    ResidualROIPoolingForward<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, bottom_data, N_ , height_, width_, bottom_mask, output_dim_, K,  top_data);
    CUDA_POST_KERNEL_CHECK;
 

  }

  template <typename Dtype>
  __global__ void ResidualROIPoolingBackwardAtomic(
    const int nthreads,
    const Dtype* top_diff,
    const int height, const int width,
    const int K,
    const int output_dim,
    Dtype* bottom_diff_data,
    Dtype* bottom_diff_mask,
    const Dtype* bottom_data,
    const Dtype* bottom_mask) {

    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int idx = index % (height * width); 
      int mask_index = index % ( K * height * width );
      int n = index / K / height / width ; 
      int k = (index / height / width)% K ;
      mask_index += n* K * height * width;


      for (int m =0 ; m < output_dim; m++){
        int data_index = idx + k*height * width + m*K*height * width + n*output_dim*K*height * width;
        bottom_diff_mask[mask_index] += top_diff[n*output_dim*K + m*K + k]*bottom_data[data_index];
        bottom_diff_data[data_index] = top_diff[n*output_dim*K + m*K + k]*(bottom_mask[mask_index] + 1.0 )/height / width ;

      }
      bottom_diff_mask[mask_index] = bottom_diff_mask[mask_index]/height / width ;
      //if(mask_index < height* width)printf("%f\t%f\n",bottom_diff_mask[mask_index],top_diff[n*output_dim*K + k]);
     }

  }

  template <typename Dtype>
  void ResidualROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    //if (!propagate_down[0]) {
    //  return;
    //}
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_mask = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff_data = bottom[0]->mutable_gpu_diff();
    Dtype* bottom_diff_mask= bottom[1]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom_diff_mask);
    caffe_gpu_set(bottom_count, Dtype(0), bottom_diff_data);
    const int count = bottom[1]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ResidualROIPoolingBackwardAtomic<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, top_diff, height_, width_, K, output_dim_, bottom_diff_data,bottom_diff_mask, bottom_data, bottom_mask);
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(ResidualROIPoolingLayer);

}  // namespace caffe
