#include <vector>

#include "caffe/layers/regularized_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {

template <typename Dtype>
void RegularizedConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    //  printf("reg_conv_forwarding\n");
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
 // printf("reg_conv_forwarding\n");
  Dtype* top_data = top[0]->mutable_gpu_data();
  for (int n = 0; n < this->num_; ++n) {
     // printf("reg_conv_forwarding\n");
      const Dtype* input = bottom_data + n * this->bottom_dim_;
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    //  printf("reg_conv_forwarding\n");
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /*  N' */, conv_out_spatial_dim_/* H' * W' */ , kernel_dim_/* C * h * w */,
        (Dtype)1., weight /*  C' * C * h * w  */, col_buffer_.gpu_data() /*  C * h * w * H' * W'   */, (Dtype)0.,  top_data + n * this->top_dim_); // C' * H' * W' 
     //   printf("reg_conv_forwarding\n");
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    //  printf("reg_conv_forwarding\n");
    }
}
template <typename Dtype>
__global__ void normalized_weights_kernel(const int n, const Dtype* weight, const int offset, Dtype* norm_val) {
  CUDA_KERNEL_LOOP(index, n) {
      norm_val[index] = 0;
      for(int i = 0; i < offset;i++){
        norm_val[index] += weight[offset + i]*weight[offset + i];
      }
  }
}
template <typename Dtype>
__global__ void div_scale_kernel(const int n, const Dtype* a,
    const Dtype b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b;
  }
}
template <typename Dtype>
__global__ void sub_diag_value(const int n, Dtype* InnerProdMat) {
  CUDA_KERNEL_LOOP(index, n) {
     InnerProdMat[index*n + index] = 0;
  }
}

template <typename Dtype>
void div_scale (int offset, const Dtype *weight , const Dtype norm_val, Dtype* out){
    div_scale_kernel<Dtype><<<CAFFE_GET_BLOCKS(offset), CAFFE_CUDA_NUM_THREADS>>>( offset, weight  ,norm_val, out);
}

template <typename Dtype>
__global__ void apply_norm_kernel(const int n/* kernel_dim * num_output*/, const Dtype* weight, const int offset, const Dtype* norm_val, Dtype * normalized_weight) {
  CUDA_KERNEL_LOOP(index, n) {
      int kernel_index = index/offset;
      normalized_weight[index] = weight[index] / norm_val[kernel_index];
  }
}
template <typename Dtype>
void RegularizedConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //    printf("reg_conv_backwarding\n");
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  //Dtype* bottom_att_diff = bottom[1]->mutable_gpu_diff();
    // Bias gradient, if necessary.
   // printf("reg_conv_backwarding\n");
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff(); 
    for (int n = 0; n < this->num_; ++n) {
      this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
    }
  }
 // printf("reg_conv_backwarding\n");
  if (this->param_propagate_down_[0] || propagate_down[0]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          const Dtype* input = bottom_data + n * this->bottom_dim_;
          
          conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
          const Dtype* col_buff = col_buffer_.gpu_data();
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ /* N' */, kernel_dim_ /* C * h * w */ , conv_out_spatial_dim_/*  H' * W'  */,
          (Dtype)1., top_diff + n * this->top_dim_ , col_buff , (Dtype)1., weight_diff);
         // printf("reg_conv_backwarding\n, %d",this->num_ );
          // adding regularized term
          // normalize weights
          normalized_weights_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_output_), CAFFE_CUDA_NUM_THREADS>>>( num_output_, weight,kernel_dim_, norm_buffer_.mutable_gpu_data());
          // apply norm
        //  printf("reg_conv_backwarding\n");
          apply_norm_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_output_), CAFFE_CUDA_NUM_THREADS>>>(num_output_, weight,kernel_dim_,  norm_buffer_.gpu_data(), norm_weight_.mutable_gpu_data()) ;
         // printf("reg_conv_backwarding\n");
          // compute innerprodmat
          caffe_gpu_gemm(CblasNoTrans, CblasTrans, num_output_, num_output_, kernel_dim_,  (Dtype)1.0,norm_weight_.gpu_data(), norm_weight_.gpu_data(), (Dtype)0., InnerProdMat_.mutable_gpu_data()); //C=alpha*A*B+beta*C
         // printf("reg_conv_backwarding\n");
          // sub diag value
          sub_diag_value<Dtype><<<CAFFE_GET_BLOCKS(num_output_), CAFFE_CUDA_NUM_THREADS>>>( num_output_,InnerProdMat_.mutable_gpu_data());
         // printf("reg_conv_backwarding, %f\n",reg_coeff_);
          //get weight diff
          caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_output_,kernel_dim_,num_output_, reg_coeff_ ,InnerProdMat_.gpu_data(), norm_weight_.gpu_data(), (Dtype)1.0, weight_diff) ;//C=alpha*A*B+beta*C
          
        }
      //  printf("reg_conv_backwarding\n");
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[0]) {
          caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_,  (Dtype)1., weight , top_diff + n * this->top_dim_ , (Dtype)0., col_buffer_.mutable_gpu_data());
          conv_col2im_gpu(col_buffer_.gpu_data(), bottom_diff + n * this->bottom_dim_);
        
        }
      //  printf("reg_conv_backwarding\n");
      }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(RegularizedConvolutionLayer);

}  // namespace caffe
