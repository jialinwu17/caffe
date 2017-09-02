#include <algorithm>
#include <vector>

#include "caffe/layers/lilac_layer.hpp"
#include <math.h>
namespace caffe {
template <typename Dtype>
void LilacLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}
template <typename Dtype>
void LilacLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i]>0? bottom_data[i]:(exp(bottom_data[i]) - 1.0);
  }
}

template <typename Dtype>
void LilacLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + exp(bottom_data[i]) * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(LilacLayer);
#endif

INSTANTIATE_CLASS(LilacLayer);
REGISTER_LAYER_CLASS(Lilac);
}  // namespace caffe
