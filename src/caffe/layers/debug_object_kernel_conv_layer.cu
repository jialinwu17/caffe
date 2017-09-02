#include <vector>

#include "caffe/layers/debug_object_kernel_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
namespace caffe {
template <typename Dtype> // n is H * W * h_samples_ * w_samples_ * C   
__global__ void object_kernel_conv_im2col_kernel(const int n, const Dtype* data_im,const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int dilation_h, const int dilation_w,
    const int h_samples, const int w_samples,const int kernel_stride,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int c_ind = index % channels;
    const int w_s_ind = index / channels % w_samples;
    const int h_s_ind = index / channels / w_samples % h_samples;
    const int w_ind = index / channels / w_samples / h_samples % width;
    const int h_ind = index / channels / w_samples / h_samples / width % height;
    const int n_ind = index / channels / w_samples / h_samples / width / height;
    const int w_sample_pad = (w_samples - 1) / 2;
    const int h_sample_pad = (h_samples - 1) / 2;
    const int b_c_ind = c_ind;
    const int b_w_ind = w_ind + (w_s_ind - w_samples + w_sample_pad + 1) * kernel_stride;
    const int b_h_ind = h_ind + (h_s_ind - h_samples + h_sample_pad + 1) * kernel_stride;
    const int top_offset = index * kernel_h * kernel_w;
    const int bottom_offset = n_ind * channels * width * height + b_c_ind * width * height +  b_h_ind * width + b_w_ind;
    const int w_pad = (kernel_w - 1) / 2;
    const int h_pad = (kernel_h - 1) / 2;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        data_col[top_offset + i * kernel_w + j] = 0;
        if ((b_h_ind + dilation_h * (i - kernel_h + 1 + h_pad ) < height) &&b_h_ind + dilation_h * (i - kernel_h + h_pad + 1) >=0){
          if ((b_w_ind + dilation_w * (j - kernel_w + w_pad + 1) < width) &&b_w_ind + dilation_w * (j - kernel_w  + w_pad + 1) >=0){
            data_col[top_offset + i * kernel_w + j] = data_im[bottom_offset + dilation_h * (i - kernel_h + 1 + h_pad ) * width + dilation_w * (j - kernel_w + 1 + w_pad)];
          }

        }
      }
    }
  }
}
template <typename Dtype>
void object_kernel_conv_im2col(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int dilation_h, const int dilation_w,
    const int h_samples, const int w_samples,const int kernel_stride,
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int pad_h = (kernel_h - 1)/2;
  int pad_w = (kernel_w - 1)/2;
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) + 1;
  int num_kernels = channels * height_col * width_col * h_samples * w_samples;
  // NOLINT_NEXT_LINE(whitespace/operators)
  object_kernel_conv_im2col_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im,channels, height, width, kernel_h, kernel_w, dilation_h, dilation_w, h_samples, w_samples,kernel_stride,data_col);
  CUDA_POST_KERNEL_CHECK;
}




template <typename Dtype>
__global__ void object_kernel_conv_col2im_kernel(const int n, const Dtype* data_col,//from HWssCkk to CHW, n is CHW
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int h_samples, const int w_samples,
    const int dilation_h, const int dilation_w, const int kernel_stride,Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {// n should be C H W
    Dtype val = 0;
    const int w_im = index % width ;
    const int h_im = (index / width) % height ;
    const int c_im = index / (width * height);
    const int h_sample_pad = (h_samples + 1)/2;
    const int w_sample_pad = (w_samples + 1)/2;
    const int h_pad = (kernel_h + 1)/2;
    const int w_pad = (kernel_w + 1)/2;
    for (int s_h = 0; s_h < h_samples; s_h ++){
      for (int s_w = 0; s_w < w_samples; s_w ++){
        for (int k_h = 0; k_h < kernel_h; k_h ++){
             for (int k_w = 0; k_w < kernel_w; k_w ++){
               int  val_ind_h = h_im + (s_h - h_samples + h_sample_pad) * kernel_stride + (k_h - kernel_h + h_pad)* dilation_h;
               int val_ind_w = w_im + (s_w - w_samples + w_sample_pad) * kernel_stride + (k_w - kernel_w + w_pad)* dilation_w;
               if (val_ind_w>=0 && val_ind_w < width && val_ind_h>=0 && val_ind_h < height){
                const int col_index = ((((h_im*width + w_im)*h_samples + s_h )*w_samples + s_w )*channels + c_im)* kernel_h*kernel_w + k_h * kernel_w  + k_w; 
                val += data_col[col_index];
               }
             }
         }
      }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
void object_kernel_conv_col2im(const Dtype* data_col, const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int h_samples,const int w_samples,const int dilation_h, const int dilation_w, const int kernel_stride,Dtype* data_im) {
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  object_kernel_conv_col2im_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, kernel_h, kernel_w, h_samples, w_samples, dilation_h, dilation_w, kernel_stride,data_im);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype> 
__device__ void gpu_matrix_mult(int nthreads,const Dtype *a,const Dtype *b, Dtype *c, int m, int n, int k,Dtype coeff){
   CUDA_KERNEL_LOOP(index,nthreads) {
   c[index] *= coeff;
   const int b_col = index % k;
   const int a_row = index / k;
   for (int i = 0 ;i< n; i++){
       c[index] +=( a[a_row * n + i]*b[i * k + b_col]);
   }
} 
}
template <typename Dtype> // n is H * W * h_samples_ * w_samples_  , weight is like  W , H, C' C, k, k, (after permuted!!), top_data N, W , H , h_samples_ * w_samples_*C'
__global__ void object_forward(const int n, const Dtype* offset_col_buffer, const Dtype* weight, const int channels, const int num_output, const int height, const int width, const int kernel_h, const int kernel_w,
    const int h_samples, const int w_samples, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, n) {
    const int w_s_ind = index % w_samples;
    const int h_s_ind = index / w_samples % h_samples;
    const int w_ind = index / w_samples / h_samples % width;
    const int h_ind = index / w_samples / h_samples / width % height;
    const int n_ind = index / w_samples / h_samples / width / height;
    
    const int offset_col_buffer_offset = index * channels * kernel_h * kernel_w;
    const int weight_offset = (n_ind * width * height + h_ind * width + w_ind) * num_output * channels*kernel_h * kernel_w;
    const int top_data_offset = (((n_ind * width * height + h_ind * width + w_ind)*h_samples + h_s_ind )* w_samples + w_s_ind)* num_output ;
   // gpu_matrix_mult<Dtype>(num_output,weight + weight_offset/*C'*C*h*w*/, offset_col_buffer + offset_col_buffer_offset/*C*h*w*H'*W'*/,  top_data + top_data_offset, num_output /*C'*/, channels * kernel_h * kernel_w/*C*h*w*/,1, (Dtype)0.); // C' * H' * W' 
    int nthreads =num_output;
    int k = 1;
    int n = channels * kernel_h * kernel_w;
    Dtype * c = top_data + top_data_offset;
    Dtype coeff = 0.0;
    const Dtype* a = weight + weight_offset;
    const Dtype* b = offset_col_buffer + offset_col_buffer_offset;
    for (int index = 0;index < nthreads;index++){
       c[index] *= coeff;
       const int b_col = index % k;
       const int a_row = index / k;
       for (int i = 0 ;i< n; i++){
           c[index] +=( a[a_row * n + i]*b[i * k + b_col]);
       }
    } 

  }
}
template <typename Dtype>
void DebugObjectKernelConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_weights = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  for (int n = 0; n < this->num_; ++n) {
      const Dtype* input = bottom_data + n * this->bottom_dim_;
      const Dtype* weights = bottom_weights + n * bottom[1]->count(1);
      Dtype* output = top_data + n * top[0]->count(1);
      object_kernel_conv_im2col(input, channels_,height_, width_, kernel_size_, kernel_size_, dilation_, dilation_, samples_, samples_, kernel_stride_,offset_col_buffer_.mutable_gpu_data());
      object_forward<Dtype><<<CAFFE_GET_BLOCKS(height_ * width_ * samples_ * samples_), CAFFE_CUDA_NUM_THREADS>>>(height_ * width_ * samples_ * samples_, offset_col_buffer_.gpu_data(), weights, channels_,num_output_,height_,width_,kernel_size_,kernel_size_,samples_,samples_,output);//N,H,W,9*C'
  }
}

template <typename Dtype> // n is H * W  , weight is like  W , H, C' C, k, k, (after permuted!!), top_data N, W , H , h_samples_ * w_samples_*C'
 __global__ void object_kernel_backward_weight(const int n, const Dtype* offset_col_buffer, const Dtype* top_diff,const int channels, const int num_output, const int height, const int width, const int kernel_h, const int kernel_w,
    const int h_samples, const int w_samples, Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    const int w_ind = index % width;
    const int h_ind = index / width % height;
    const int kernel_dim_ = channels * kernel_h*kernel_w;
    for(int s_h = 0; s_h < h_samples;s_h ++){
      for(int s_w = 0; s_w < w_samples; s_w++){
        int buffer_offset = ((index*h_samples + s_h)*w_samples)* channels*kernel_h*kernel_w;
        int top_offset =  ((index*h_samples + s_h)*w_samples)*num_output;
        int weight_offset = index*num_output*channels*kernel_h*kernel_w;
        //gpu_matrix_mult<Dtype>( num_output*kernel_dim_, top_diff + top_offset ,offset_col_buffer + buffer_offset , weight_diff + weight_offset, num_output /* C' */, 1,kernel_dim_ /* C * h * w */ ,(Dtype)1.);
        int nthreads =num_output*kernel_dim_ ;
        int k = kernel_dim_;
        int n = 1;
        Dtype * c = weight_diff + weight_offset;
        Dtype coeff = 1.0;
        const Dtype* a = top_diff + top_offset;
        const Dtype* b = offset_col_buffer + buffer_offset;
        for (int index = 0;index < nthreads;index++){
           c[index] *= coeff;
           const int b_col = index % k;
           const int a_row = index / k;
           for (int i = 0 ;i< n; i++){
               c[index] +=( a[a_row * n + i]*b[i * k + b_col]);
           }
        } 

      }
    }
  }
}

template <typename Dtype> // n is H * W  , weight is like  W , H, C' C, k, k, (after permuted!!), top_data N, W , H , h_samples_ * w_samples_*C'
__global__ void object_kernel_backward_data(const int n, const Dtype* weight, const Dtype* top_diff,const int channels, const int num_output, const int height, const int width, const int kernel_h, const int kernel_w,
    const int h_samples, const int w_samples, Dtype* offset_col_buffer) {
  CUDA_KERNEL_LOOP(index, n) {
    const int w_ind = index % width;
    const int h_ind = index / width % height;
    const int kernel_dim_ = channels * kernel_h*kernel_w;
    for(int s_h = 0; s_h < h_samples;s_h ++){
      for(int s_w = 0; s_w < w_samples; s_w++){
        int buffer_offset = ((index*h_samples + s_h)*w_samples)* channels*kernel_h*kernel_w;
        int top_offset =  ((index*h_samples + s_h)*w_samples)*num_output;
        int weight_offset = index*num_output*channels*kernel_h*kernel_w;
        //gpu_matrix_mult<Dtype>( kernel_dim_, top_diff + top_offset , weight + weight_offset , offset_col_buffer + buffer_offset , 1 /* C' */, num_output, kernel_dim_ /* C * h * w */ , (Dtype)1. );
       int nthreads =kernel_dim_;
        int k = kernel_dim_;
        int n = num_output;
        Dtype * c =offset_col_buffer + buffer_offset;
        Dtype coeff = 1.0;
        const Dtype* a = top_diff + top_offset;
        const Dtype* b =weight + weight_offset ;
        for (int index = 0;index < nthreads;index++){
           c[index] *= coeff;
           const int b_col = index % k;
           const int a_row = index / k;
           for (int i = 0 ;i< n; i++){
               c[index] +=( a[a_row * n + i]*b[i * k + b_col]);
           }
        } 







      }
    }
  }
}
template <typename Dtype>
void DebugObjectKernelConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_weights = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_weight_diff = bottom[1]->mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), Dtype(0.0), bottom_diff);
  caffe_gpu_set(bottom[1]->count(), Dtype(0.0), bottom_weight_diff);
  for (int n = 0; n < this->num_; ++n) {
    // gradient w.r.t. weight. Note that we will accumulate diffs.
    const Dtype* weights = bottom_weights + n * bottom[1]->count(1);
    const Dtype* diff = top_diff + n * top[0]->count(1);
    const Dtype* input = bottom_data + n * this->bottom_dim_;
    Dtype* weight_diff = bottom_weight_diff + n * bottom[1]->count(1);
    object_kernel_conv_im2col(input, channels_,height_, width_, kernel_size_, kernel_size_, dilation_, dilation_, samples_, samples_,  kernel_stride_,offset_col_buffer_.mutable_gpu_data());
    const Dtype* offset_col_buff = offset_col_buffer_.gpu_data();
    object_kernel_backward_weight<<<CAFFE_GET_BLOCKS(height_ * width_ ), CAFFE_CUDA_NUM_THREADS>>>(height_*width_ , offset_col_buff, diff, channels_, num_output_, height_, width_, kernel_size_, kernel_size_, samples_, samples_, weight_diff);
    
    // gradient w.r.t. bottom data, if necessary.
    Dtype * att_col_diff_buff = offset_col_buffer_.mutable_gpu_data();
    object_kernel_backward_data<<<CAFFE_GET_BLOCKS(height_ * width_ ), CAFFE_CUDA_NUM_THREADS>>>(height_*width_ , weights, diff, channels_, num_output_, height_, width_, kernel_size_, kernel_size_, samples_, samples_, att_col_diff_buff);
    object_kernel_conv_col2im(offset_col_buffer_.gpu_data(), channels_, height_, width_,  kernel_size_, kernel_size_, samples_,samples_, dilation_, dilation_,kernel_stride_, bottom_diff + n*bottom[0]->count(1));
  }
  caffe_gpu_scal(bottom[0]->count(),Dtype(lr_mul),bottom_diff);
  //caffe_gpu_scal(bottom[1]->count(),Dtype(lr_mul),bottom_weight_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(DebugObjectKernelConvolutionLayer);

}  // namespace caffe