// --------------------------------------------------------
// R-FCN
// Written by Yi Li, 2016.
// --------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/triangle_pooling_layer.hpp"
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {

  template <typename Dtype>
  __global__ void PSROIPoolingForward(
    const int nthreads,
    const Dtype* bottom_data,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois,
    const int output_dim,
    const int group_size,
    Dtype* top_data,
    int* mapping_channel, Dtype* tmp1,Dtype* tmp2) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w =
        static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
      Dtype roi_start_h =
        static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
      Dtype roi_end_w =
        static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
      Dtype roi_end_h =
        static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      int hstart = floor(static_cast<Dtype>(ph) * bin_size_h
                          + roi_start_h);
      int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
                          + roi_start_w);
      int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
                        + roi_start_h);
      int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
                        + roi_start_w);
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0), width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      int gw = pw;
      int gh = ph;
      int c1 = (ctop*group_size + gh)*group_size + gw;
      int c2 = c1 + output_dim * group_size * group_size;
      const Dtype* bottom_data1 = bottom_data + (roi_batch_ind * channels + c1) * height * width;
      const Dtype* bottom_data2 = bottom_data + (roi_batch_ind * channels + c2) * height * width;
      Dtype out_sum1 = 0.0;
      Dtype out_sum2 = 0.0;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          int bottom_index = h*width + w;
          Dtype ratio1 = (Dtype)( h - hstart) / (Dtype)(  w - wstart) ;
          Dtype ratio2 = (Dtype)( hend - hstart) / (Dtype)(  wend - wstart) ;
          if( ratio1 >= ratio2){
            out_sum1 += bottom_data1[bottom_index];
            tmp1[index] += 1.0;
          }
          if ( ratio1 <= ratio2){
            out_sum2 += bottom_data2[bottom_index];
            tmp2[index] += 1.0;
          } 
        }
      }

      //Dtype bin_area = (hend - hstart)*(wend - wstart);
      if (tmp1[index] == 0 && tmp2[index] == 0){
          top_data[index]  = 0.0;
      }
      else if(tmp1[index] == 0 &&tmp2[index] > 0 ){
           top_data[index] = out_sum2/tmp2[index];
      }
      else if(tmp1[index] > 0 &&tmp2[index] == 0 ){
           top_data[index] = out_sum1/tmp1[index];
      }
      else{
           top_data[index] = 0.5 * (out_sum1/tmp1[index] + out_sum2/tmp2[index]);
      }
      //printf("index:%d tmp1 %.2f: tmp2:%.2f top_data: %.6f\n",index,tmp1[index],tmp2[index],top_data[index] );
      mapping_channel[index] = c1;
    }
  }

  template <typename Dtype>
  void TrianglePoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int* mapping_channel_ptr = mapping_channel_.mutable_gpu_data();
    Dtype* tmp1 = tmp1_.mutable_gpu_data();
    Dtype* tmp2 = tmp2_.mutable_gpu_data();
    int count = top[0]->count();
    //printf("%d\n",count);
    caffe_gpu_set(count, Dtype(0), top_data);
    caffe_gpu_set(count, -1, mapping_channel_ptr);
    caffe_gpu_set(count, Dtype(0), tmp1);
    caffe_gpu_set(count, Dtype(0), tmp2);
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSROIPoolingForward<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, bottom_data, spatial_scale_,
      channels_, height_, width_, pooled_height_,
      pooled_width_, bottom_rois, output_dim_, group_size_,
      top_data, mapping_channel_ptr,tmp1,tmp2);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  __global__ void PSROIPoolingBackwardAtomic(
    const int nthreads,
    const Dtype* top_diff,
    const int* mapping_channel,
    const int num_rois,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim,
    Dtype* bottom_diff,
    const Dtype* bottom_rois,
    const Dtype* tmp1,
    const Dtype* tmp2) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w =
        static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
      Dtype roi_start_h =
        static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
      Dtype roi_end_w =
        static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
      Dtype roi_end_h =
        static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      int hstart = floor(static_cast<Dtype>(ph)* bin_size_h
        + roi_start_h);
      int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
        + roi_start_w);
      int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
        + roi_start_h);
      int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
        + roi_start_w);
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0), width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // Compute c at bottom
      int c1 = mapping_channel[index];
      int c2 = c1 + output_dim * pooled_height * pooled_height;
      Dtype* offset_bottom_diff1 = bottom_diff + (roi_batch_ind * channels + c1) * height * width;
      Dtype* offset_bottom_diff2 = bottom_diff + (roi_batch_ind * channels + c2) * height * width;
      //Dtype bin_area = (hend - hstart)*(wend - wstart);
      Dtype diff_val1 = 0.0;
      Dtype diff_val2 = 0.0;
      if (!is_empty){
        if (tmp1[index] == 0 && tmp2[index] == 0){
          diff_val1  = 0.0;
          diff_val2  = 0.0;
        }
        else if(tmp1[index] == 0 &&tmp2[index] > 0 ){
             diff_val2  = 0.5 * top_diff[index] / tmp2[index];
        }
        else if(tmp1[index] > 0 &&tmp2[index] == 0 ){
             diff_val1  = 0.5 * top_diff[index] / tmp1[index];
        }
        else{
             diff_val2  = 0.5 * top_diff[index] / tmp2[index];
             diff_val1  = 0.5 * top_diff[index] / tmp1[index];
            // printf("%.4f\t%.4f\n",tmp1[index],tmp2[index]);
        }
      }
      //Dtype diff_val1 = is_empty ? 0. : top_diff[index] * 0.5 / bin_area;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          int bottom_index = h*width + w;
          Dtype ratio1 = (Dtype)( h - hstart) / (Dtype)(  w - wstart) ;
          Dtype ratio2 = (Dtype)( hend - hstart) / (Dtype)(  wend - wstart) ;
          if( ratio1 >= ratio2){
            //offset_bottom_diff1[bottom_index] += top_diff[index];
            caffe_gpu_atomic_add(diff_val1, offset_bottom_diff1 + bottom_index);
          }
          if ( ratio1 <= ratio2){
            caffe_gpu_atomic_add(diff_val2, offset_bottom_diff2 + bottom_index);
          } 
          //caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
          //printf("%f\n",*(offset_bottom_diff + bottom_index));
        }
      }
      
    }
  }

  template <typename Dtype>
  void TrianglePoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
      return;
    }

    const Dtype* bottom_rois = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    const int* mapping_channel_ptr = mapping_channel_.gpu_data();
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
    caffe_gpu_set(bottom_count, Dtype(0), bottom_diff);
    const int count = top[0]->count();
    const Dtype* tmp1 = tmp1_.gpu_data();
    const Dtype* tmp2 = tmp2_.gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSROIPoolingBackwardAtomic<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, top_diff, mapping_channel_ptr,
      top[0]->num(), spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, output_dim_, bottom_diff,
      bottom_rois,tmp1,tmp2);
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(TrianglePoolingLayer);

}  // namespace caffe
