#include <vector>

#include "caffe/layers/attention_random_input_conv_layer.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
namespace caffe {
template <typename Dtype>
__global__ void inspect_random_kernel(const int n, Dtype* a) {
  CUDA_KERNEL_LOOP(index, n) {
      printf("%f\n",a[index] );
  }
}
template <typename Dtype>
__global__ void add_number_kernel(const int n, const Dtype* a, const Dtype b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b;
  }
}
template <typename Dtype>
__global__ void mul_number_kernel(const int n, const Dtype* a, const Dtype b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b;
  }
}

template <typename Dtype>
void AttentionRandomInputConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_attention = bottom[1]->gpu_data();
  const Dtype* random = bottom[2]->mutable_gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  //printf("forward attention conv layer 0\n");
  //srand((unsigned int)time(NULL));
  for (int n = 0; n < this->num_; ++n) {
      const Dtype* input = bottom_data + n * this->bottom_dim_;
      const Dtype* att_input = bottom_attention + n *  kernel_dim_ * height_ * width_;
      const Dtype* random_input = random + n *  kernel_dim_ * height_ * width_;
  //    printf("forward attention conv layer 1\n");
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
  //    printf("forward attention conv layer 2\n");
      caffe_gpu_mul( kernel_dim_ * height_ * width_,col_buffer_.gpu_data(),att_input, attention_col_buffer_.mutable_gpu_data());
  //    printf("forward attention conv layer 3\n");
  
    //printf("lilac!!\n");
    //int number = kernel_dim_ * height_ * width_;
    //caffe_gpu_rng_uniform(kernel_dim_ * height_ * width_,mini, maxi,random_input);
    //printf("lilac!!\n");
    //inspect_random_kernel<<<CAFFE_GET_BLOCKS(10), CAFFE_CUDA_NUM_THREADS>>>(10,random_input);
    //printf("\n");
    //mul_number_kernel<Dtype><<<CAFFE_GET_BLOCKS(kernel_dim_ * height_ * width_), CAFFE_CUDA_NUM_THREADS>>>(kernel_dim_ * height_ * width_, random_input, maxi - mini, random_input);
    //add_number_kernel<Dtype><<<CAFFE_GET_BLOCKS(kernel_dim_ * height_ * width_), CAFFE_CUDA_NUM_THREADS>>>(kernel_dim_ * height_ * width_, random_input, mini, random_input);
  
      caffe_gpu_mul(kernel_dim_ * height_ * width_ ,attention_col_buffer_.gpu_data(), random_input, attention_col_buffer_.mutable_gpu_data());

      const Dtype* att_col_buff = attention_col_buffer_.gpu_data();
  //    printf("forward attention conv layer 4\n");
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /*  N' */, conv_out_spatial_dim_/* H' * W' */ , kernel_dim_/* C * h * w */,
        (Dtype)1., weight /*  C' * C * h * w  */, att_col_buff /*  C * h * w * H' * W'   */, (Dtype)0.,  top_data + n * this->top_dim_); // C' * H' * W' 
  //    printf("forward attention conv layer 5\n");

      if (this->bias_term_) {
   //     printf("forward attention conv layer 5.5\n");
        const Dtype* bias = this->blobs_[1]->gpu_data();
     //   printf("forward attention conv layer 6\n");
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
       // printf("forward attention conv layer 7\n");
      }
    }
}

template <typename Dtype>
void AttentionRandomInputConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_attention = bottom[1]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_att_diff = bottom[1]->mutable_gpu_diff();
  const Dtype* random = bottom[2]->gpu_data();
    // Bias gradient, if necessary.
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    for (int n = 0; n < this->num_; ++n) {
      this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
    }
  }
  if (this->param_propagate_down_[0] || propagate_down[0]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        const Dtype* att_input = bottom_attention + n * kernel_dim_ * height_ * width_;
        const Dtype* random_input = random + n *  kernel_dim_ * height_ * width_;
        if (this->param_propagate_down_[0]) {
          const Dtype* input = bottom_data + n * this->bottom_dim_;
          
          conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
          const Dtype* col_buff = col_buffer_.gpu_data();
          caffe_gpu_mul(kernel_dim_ * height_ * width_,col_buff,att_input, attention_col_buffer_.mutable_gpu_data());
          caffe_gpu_mul( kernel_dim_ * height_ * width_,random_input,attention_col_buffer_.gpu_data(), attention_col_buffer_.mutable_gpu_data());
          const Dtype* att_col_buff = attention_col_buffer_.gpu_data();
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ /* N' */, kernel_dim_ /* C * h * w */ , conv_out_spatial_dim_/*  H' * W'  */,
          (Dtype)1., top_diff + n * this->top_dim_ , att_col_buff, (Dtype)1., weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[0]) {
          Dtype * att_col_diff_buff = attention_col_diff_buff_.mutable_gpu_data();
          caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_,  (Dtype)1., weight , top_diff + n * this->top_dim_ , (Dtype)0., att_col_diff_buff );
          caffe_gpu_mul(kernel_dim_ * height_ * width_, attention_col_diff_buff_.gpu_data() ,random_input, attention_col_diff_buff_.mutable_gpu_data());
          caffe_gpu_mul(kernel_dim_ * height_ * width_, attention_col_diff_buff_.gpu_data() ,att_input, col_diff_buffer_.mutable_gpu_data());
        
        conv_col2im_gpu(col_diff_buffer_.gpu_data(), bottom_diff + n * this->bottom_dim_);
        caffe_gpu_mul(kernel_dim_ * height_ * width_, attention_col_diff_buff_.gpu_data() ,col_buffer_.gpu_data(),bottom_att_diff + n * kernel_dim_ * height_ * width_);
        }
      }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(AttentionRandomInputConvolutionLayer);

}  // namespace caffe
