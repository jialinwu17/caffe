#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/resize_layer.hpp"


using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void ROIPoolForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    Dtype roi_start_w = 0.0; roi_start_w *= spatial_scale;
    Dtype roi_start_h = 0.0;roi_start_h*= spatial_scale;
    Dtype roi_end_w = static_cast<Dtype>(width); roi_end_w*= spatial_scale;
    Dtype roi_end_h = static_cast<Dtype>(height); roi_end_h*= spatial_scale;
    const Dtype* bottom_data_cur = bottom_data + (n*channels+c)* height * width;
    // Force malformed ROIs to be 1x1
    Dtype roi_width = max(roi_end_w - roi_start_w , 1.0);
    Dtype roi_height = max(roi_end_h - roi_start_h , 1.0);
    Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);
    Dtype h_cur = roi_start_h + static_cast<Dtype>(ph) * bin_size_h;
    Dtype w_cur = roi_start_w + static_cast<Dtype>(pw) * bin_size_w;   
    /*
    //int hstart = static_cast<int>(floor( h_cur));
    //int wstart = static_cast<int>(floor(w_cur));
    //int hend = static_cast<int>(ceil(h_cur));
    //int wend = static_cast<int>(ceil(w_cur));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(static_cast<Dtype>(hstart ), 0.0), static_cast<Dtype>(height));
    hend = min(max(static_cast<Dtype>(hend) , 0.0), static_cast<Dtype>(height));
    wstart = min(max(static_cast<Dtype>(wstart), 0.0), static_cast<Dtype>(width));
    wend = min(max(static_cast<Dtype>(wend),0.0),static_cast<Dtype>( width));
    bool is_empty = (hend < hstart) || (wend < wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    Dtype u = h_cur - hstart;
    Dtype v = w_cur - wstart;
    bottom_data += (c) * height * width;
    int bottom_index1 = hstart* width + wstart;
    int bottom_index2 = hstart* width + wend;
    int bottom_index3 = hend* width + wstart;
    int bottom_index4 = hend* width + wend;
    */
    int hend = ceil(h_cur - 0.5);
    //int pwstart = floor(pw_cur);
    int wend = ceil(w_cur - 0.5);
    //phstart = min(max(static_cast<Dtype>(phstart), 0.0), static_cast<Dtype>(pooled_height - 1.0));
    hend = min(max(static_cast<Dtype>(hend), 1.0), static_cast<Dtype>(height - 1.0));
    //pwstart = min(max(static_cast<Dtype>(pwstart), 0.0), static_cast<Dtype>(pooled_width - 1.0));
    wend = min(max(static_cast<Dtype>(wend), 1.0), static_cast<Dtype>(width - 1.0));
    int hstart = hend - 1;
    int wstart = wend - 1;
    int bottom_index1 = hstart* width + wstart;
    int bottom_index2 = hstart* width + wend;
    int bottom_index3 = hend* width + wstart;
    int bottom_index4 = hend* width + wend;
    Dtype u = h_cur - hstart - 0.5 ;
    Dtype v = w_cur - wstart - 0.5 ;
    //printf("%d,%d\n",bottom_index1,bottom_index4);


    top_data[index] = (1 - u)*(1 - v)*bottom_data_cur[bottom_index1] + (1 - u)*(v)*bottom_data_cur[bottom_index2]+(u)*(1 - v)*bottom_data_cur[bottom_index3] + u*v*bottom_data_cur[bottom_index4];
  }
}

template <typename Dtype>
void ResizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//  printf("resize Forward_gpu\n");
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
 // printf("resize Forward_gpu\n");
  int count = top[0]->count();
  printf("%d\n",count);
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_,  top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ROIPoolBackward(const int nthreads, const Dtype* top_diff,
    const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      Dtype roi_start_w = 0.0; roi_start_w *= spatial_scale;
      Dtype roi_start_h = 0.0;roi_start_h*= spatial_scale;
      Dtype roi_end_w = static_cast<Dtype>(width); roi_end_w*= spatial_scale;
      Dtype roi_end_h = static_cast<Dtype>(height); roi_end_h*= spatial_scale;

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      
      // Force malformed ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w , 1.0);
      Dtype roi_height = max(roi_end_h - roi_start_h , 1.0);

      Dtype bin_size_h = static_cast<Dtype>(roi_height)
                         / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)
                         / static_cast<Dtype>(pooled_width);
      Dtype ph_cur = static_cast<Dtype>(h - roi_start_h) / bin_size_h;
      Dtype pw_cur = static_cast<Dtype>(w - roi_start_w) / bin_size_w; 
      //int phstart = floor(ph_cur);
      int phend = ceil(ph_cur - 0.5);
      //int pwstart = floor(pw_cur);
      int pwend = ceil(pw_cur - 0.5);

      //phstart = min(max(static_cast<Dtype>(phstart), 0.0), static_cast<Dtype>(pooled_height - 1.0));
      phend = min(max(static_cast<Dtype>(phend), 1.0), static_cast<Dtype>(pooled_height - 1.0));
      //pwstart = min(max(static_cast<Dtype>(pwstart), 0.0), static_cast<Dtype>(pooled_width - 1.0));
      pwend = min(max(static_cast<Dtype>(pwend), 0.0), static_cast<Dtype>(pooled_width - 1.0));
      int phstart = phend - 1;
      int pwstart = pwend - 1;

      int top_index1 = phstart* pooled_width + pwstart;
      int top_index2 = phstart* pooled_width + pwend;
      int top_index3 = phend* pooled_width + pwstart;
      int top_index4 = phend* pooled_width + pwend;
      Dtype u = ph_cur - phstart - 0.5 ;
      Dtype v = pw_cur - pwstart - 0.5 ;
      bottom_diff[index] += (1 - u)*(1 - v)*offset_top_diff[top_index1] + (1 - u)*(v)*offset_top_diff[top_index2]+(u)*(1 - v)*offset_top_diff[top_index3] + u*v*offset_top_diff[top_index4];
     }
  }
}

template <typename Dtype>
void ResizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
 // printf("resize back \n");
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, top[0]->num(), spatial_scale_, channels_,
      height_, width_, pooled_height_, pooled_width_, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
//  printf("resize back \n");
}

INSTANTIATE_LAYER_GPU_FUNCS(ResizeLayer);

}  // namespace caffe
