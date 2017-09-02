#include <vector>

#include "caffe/layers/cross_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void CrossConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = bottom[1]->gpu_data();
  int offset = bottom[1]->count(1);
  //printf("%d %d: %d %d %d %d\n",bottom[1]->count(0),bottom[1]->shape(0),bottom[1]->shape(1),bottom[1]->shape(2),bottom[1]->shape(3),bottom[1]->shape(4));
  for (int i = 0; i < 1/*bottom.size()*/; ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight + n * offset,
          top_data + n * this->top_dim_);
      /*if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }*/
    }
  }
}

template <typename Dtype>
void CrossConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = bottom[1]->gpu_data();
  Dtype* weight_diff = bottom[1]->mutable_gpu_diff();
  int offset = bottom[1]->count(1);
//  printf("cross conv back\n");
   const Dtype* top_diff = top[0]->gpu_diff();
   caffe_gpu_set(bottom[1]->count(),Dtype(0.0),weight_diff );
   const Dtype* bottom_data = bottom[0]->gpu_data();
   Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  //    printf("cross conv back\n");
   for (int n = 0; n < this->num_; ++n) {
     // gradient w.r.t. weight. Note that we will accumulate diffs.
      this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_, top_diff + n * this->top_dim_, weight_diff + n *offset);
      this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight + n *offset, bottom_diff + n * this->bottom_dim_);
        
   }
 }

INSTANTIATE_LAYER_GPU_FUNCS(CrossConvolutionLayer);

}  // namespace caffe
