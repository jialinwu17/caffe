#ifndef CAFFE_DEBUG_OBJECT_KERNEL_CONV_LAYER_HPP_
#define CAFFE_DEBUG_OBJECT_KERNEL_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

/**
 * @brief Convolves the input image with a bank of learned filters,
 *        and (optionally) adds biases.
 *
 *   Caffe convolves by reduction to matrix multiplication. This achieves
 *   high-throughput and generality of input and filter dimensions but comes at
 *   the cost of memory for matrices. This makes use of efficiency in BLAS.
 *
 *   The input is "im2col" transformed to a channel K' x H x W data matrix
 *   for multiplication with the N x K' x H x W filter matrix to yield a
 *   N' x H x W output matrix that is then "col2im" restored. K' is the
 *   input channel * kernel height * kernel width dimension of the unrolled
 *   inputs so that the im2col matrix has a column for each input region to
 *   be filtered. col2im restores the output spatial structure by rolling up
 *   the output channel N' columns of the output matrix.
 */
template <typename Dtype>
class DebugObjectKernelConvolutionLayer : public Layer<Dtype> {
 public:
  
  explicit DebugObjectKernelConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);
  void forward_gpu_bias(Dtype* bias, const Dtype* input);
  void forward_cpu_bias(Dtype* bias, const Dtype* input);
  virtual inline const char* type() const { return "DebugObjectKernelConvolution"; }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  
  virtual void compute_output_shape();
 
  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  // Compute height_out_ and width_out_ from other parameters.

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  int stride_;
  /// @brief The spatial dimensions of the padding.
  int kernel_size_;
  /// @brief The spatial dimensions of the dilation.
  int dilation_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> offset_col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;
  int height_;
  Dtype lr_mul;
  int width_;
  int channel_axis_;
  int num_;
  int pad_;
  int channels_;
  int group_;
  int out_spatial_dim_;
  int weight_offset_;
  int num_output_;
  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int samples_;
  int kernel_stride_;
  int col_offset_;
  int output_offset_;
  Blob<Dtype> offset_col_buffer_;
};

}  // namespace caffe

#endif  // CAFFE_CONV_LAYER_HPP_