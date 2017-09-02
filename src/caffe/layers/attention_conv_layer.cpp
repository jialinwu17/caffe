#include <vector>
#include "caffe/layers/attention_conv_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AttentionConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    kernel_shape_data[0] = conv_param.kernel_h();//3
    kernel_shape_data[1] = conv_param.kernel_w();//3
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    for (int i = 0; i < num_spatial_axes_; ++i) {
      kernel_shape_data[i] = conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
    }
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=  kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  conv_out_channels_ = num_output_;
  conv_in_channels_ = channels_;
  
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1); // C * h * w
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_; // N' * C * h * w / g
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void AttentionConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  width_ = bottom[0]->shape(3);
  height_ = bottom[0]->shape(2);
  //printf("height: %d, width:%d\n", height_,width_);
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  
  conv_out_spatial_dim_ = top[0]->count(first_spatial_axis); // H' * W' 
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
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
  col_buffer_shape_.clear();
 // printf("kernel_dim_ : %d\n",kernel_dim_ * group_);
  col_buffer_shape_.push_back(kernel_dim_ * group_); //C * h * w
  for (int i = 0; i < num_spatial_axes_; ++i) {
     col_buffer_shape_.push_back(output_shape_[i]);//C * h * w * H' * W' 
  }
 // printf("kernel_dim_ : %d,output_shape_0: %d,output_shape_1: %d \n",kernel_dim_ * group_,output_shape_[0],output_shape_[1]);
  col_buffer_.Reshape(col_buffer_shape_);
  col_diff_buffer_.Reshape(col_buffer_shape_);
  attention_col_buffer_.Reshape(col_buffer_shape_);
  attention_col_diff_buff_.Reshape(col_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void AttentionConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void AttentionConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // printf("attention conv forward 0\n");
    const Dtype* weight = this->blobs_[0]->cpu_data();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_attention = bottom[1]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
  //  printf("attention conv forward 0\n");
    for (int n = 0; n < this->num_; ++n) { 
      //this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
      //    top_data + n * this->top_dim_);
   //   printf("attention conv forward 1\n");
      const Dtype* input = bottom_data + n * this->bottom_dim_;
      const Dtype* att_input = bottom_attention + n * kernel_dim_ * height_ * width_;
   //   printf("attention conv forward 2\n");
      if (!is_1x1_) {
          conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
      }
   //   printf("attention conv forward 3\n");
      const Dtype* col_buff = col_buffer_.cpu_data();
      caffe_mul( kernel_dim_ * height_ * width_,col_buff,att_input, attention_col_buffer_.mutable_cpu_data());
  //    printf("attention conv forward 4\n");
      const Dtype* att_col_buff = attention_col_buffer_.cpu_data();
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /*  N' */, conv_out_spatial_dim_/* H' * W' */ , kernel_dim_/* C * h * w */,
        (Dtype)1., weight /*  C' * C * h * w  */, att_col_buff /*  C * h * w * H' * W'   */, (Dtype)0.,  top_data + n * this->top_dim_); // C' * H' * W' 
   //   printf("attention conv forward 5\n");
      if (this->bias_term_) {
   //   printf("attention conv forward 6\n");
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
   //     printf("attention conv forward 7\n");
      }
  }
}

template <typename Dtype>
void AttentionConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_attention = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype* bottom_att_diff = bottom[1]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[0]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        const Dtype* att_input = bottom_attention + n * kernel_dim_ * height_ * width_;
        if (this->param_propagate_down_[0]) {
          const Dtype* input = bottom_data + n * this->bottom_dim_;
          
          conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
          const Dtype* col_buff = col_buffer_.cpu_data();
          caffe_mul( kernel_dim_ * height_ * width_,col_buff,att_input, attention_col_buffer_.mutable_cpu_data());
          const Dtype* att_col_buff = attention_col_buffer_.cpu_data();
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ /* N' */, kernel_dim_ /* C * h * w */ , conv_out_spatial_dim_/*  H' * W'  */,
         (Dtype)1., top_diff + n * this->top_dim_ , att_col_buff, (Dtype)1., weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[0]) {
         // this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
         //     bottom_diff + n * this->bottom_dim_);
         Dtype* att_col_diff_buff = attention_col_diff_buff_.mutable_cpu_data();
          caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_, conv_out_channels_,  (Dtype)1., weight , top_diff + n * this->top_dim_ ,(Dtype)0., att_col_diff_buff );
        caffe_mul(kernel_dim_ * height_ * width_, attention_col_diff_buff_.cpu_data() ,att_input, col_diff_buffer_.mutable_cpu_data());
        conv_col2im_cpu(col_diff_buffer_.cpu_data(), bottom_diff + n * this->bottom_dim_);
        caffe_mul( kernel_dim_ * height_ * width_, attention_col_diff_buff_.cpu_data() ,col_buffer_.cpu_data(),bottom_att_diff + n  * kernel_dim_ * height_ * width_);
        }
      }
    }
  
}
template <typename Dtype>
void AttentionConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}
template <typename Dtype>
void AttentionConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_ /* N' */,
      out_spatial_dim_/* H' * W' */, 1, (Dtype)1., bias/* 1 * N' */, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}
template <typename Dtype>
void AttentionConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}
template <typename Dtype>
void AttentionConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifdef CPU_ONLY
STUB_GPU(AttentionConvolutionLayer);
#endif

INSTANTIATE_CLASS(AttentionConvolutionLayer);
REGISTER_LAYER_CLASS(AttentionConvolution);

}  // namespace caffe
