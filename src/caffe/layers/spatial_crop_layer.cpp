#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/layers/spatial_crop_layer.hpp"
#include "caffe/net.hpp"


namespace caffe {

template <typename Dtype>
void SpatialCropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // LayerSetup() handles the number of dimensions; Reshape() handles the sizes.
  // bottom[0] supplies the data
  // bottom[1] supplies the size
  const SpatialCropParameter& param = this->layer_param_.spatial_crop_param();
  int input_dim = bottom[0]->num_axes();
  const int start_axis = bottom[0]->CanonicalAxisIndex(param.axis());
  num_crops = param.num_crops();
}

template <typename Dtype>
void SpatialCropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const SpatialCropParameter& param = this->layer_param_.spatial_crop_param();
  int input_dim = bottom[0]->num_axes();

  // Initialize offsets to 0 and the new shape to the current shape of the data.
  vector<int> new_shape(bottom[0]->shape());
  index_map_.Reshape(new_shape);
  width_ = new_shape[3];
  height_ = new_shape[2];
  channels_ = new_shape[1];
  new_shape[1] *= num_crops * num_crops;
  new_shape[2] /=  num_crops;
  new_shape[3] /=  num_crops;
  top[0]->Reshape(new_shape);
  
  copy_h = new_shape[2];
  copy_w = new_shape[3];
}

template <typename Dtype>
void SpatialCropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
/*  std::vector<int> indices(top[0]->num_axes(), 0);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  crop_copy(bottom, top, offsets, indices, 0, bottom_data, top_data, true);*/
}

template <typename Dtype>
void SpatialCropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
 /* const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    std::vector<int> indices(top[0]->num_axes(), 0);
    crop_copy(bottom, top, offsets, indices, 0, top_diff, bottom_diff, false);
  }*/
}

#ifdef CPU_ONLY
STUB_GPU(SpatialCropLayer);
#endif

INSTANTIATE_CLASS(SpatialCropLayer);
REGISTER_LAYER_CLASS(SpatialCrop);

}  // namespace caffe
