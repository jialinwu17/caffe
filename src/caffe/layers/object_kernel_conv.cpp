#include <vector>
#include "caffe/layers/object_kernel_conv_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ObjectKernelConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ObjectKernelConvolutionParameter conv_param = this->layer_param_.object_kernel_convolution_param();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  stride_ = conv_param.stride();
  kernel_size_ = conv_param.kernel_size();
  dilation_ = conv_param.dilation();
  channels_ = bottom[0]->shape(channel_axis_);
  printf("haha\n");
  num_output_ = this->layer_param_.object_kernel_convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  printf("haha\n");
  conv_out_channels_ = num_output_;
  conv_in_channels_ = channels_;
  kernel_dim_ = channels_ * kernel_size_ * kernel_size_; // C * h * w
  // Propagate gradients to the parameters (as directed by backward pass).
  printf("haha\n");
  samples_ = conv_param.samples();
  kernel_stride_ = conv_param.kernel_stride();
  pad_ = (kernel_size_ - 1)/2;
  printf("haha\n");
  
}

template <typename Dtype>
void ObjectKernelConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;

  num_ = bottom[0]->count(0, channel_axis_);
  bottom_shape_ = &bottom[0]->shape();
  width_ = bottom[0]->shape(3);
  height_ = bottom[0]->shape(2);
  compute_output_shape();

  vector<int> top_shape(bottom[0]->shape().begin(), bottom[0]->shape().begin() + channel_axis_);
  
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  
  top[0]->Reshape(top_shape);
  conv_out_spatial_dim_ = top[0]->count(first_spatial_axis); // H' * W' 
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;//C * k * k * H * W
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ ;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
  }

  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  
  offset_col_buffer_shape_.clear();
  
  //col_buffer_shape_.clear();
  //col_buffer_shape_.push_back(kernel_dim_ * group_); //C * h * w
  for (int i = 0; i < num_spatial_axes_; ++i) {
     //col_buffer_shape_.push_back(output_shape_[i]);//C * h * w * H' * W' 
     offset_col_buffer_shape_.push_back(output_shape_[i]);
  }

  offset_col_buffer_shape_.push_back(samples_);
  offset_col_buffer_shape_.push_back(samples_);
  offset_col_buffer_shape_.push_back(kernel_dim_);
  offset_col_buffer_.Reshape(offset_col_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);

}

template <typename Dtype>
void ObjectKernelConvolutionLayer<Dtype>::compute_output_shape() {

  this->output_shape_.clear();

  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis

    const int input_dim = this->input_shape(i + 1);

    const int kernel_extent = dilation_ * (kernel_size_ - 1) + 1;
   
    const int output_dim = (input_dim + 2 * pad_ - kernel_extent) + 1;

    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ObjectKernelConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void ObjectKernelConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
  
}

#ifdef CPU_ONLY
STUB_GPU(ObjectKernelConvolutionLayer);
#endif

INSTANTIATE_CLASS(ObjectKernelConvolutionLayer);
REGISTER_LAYER_CLASS(ObjectKernelConvolution);

}  // namespace caffe