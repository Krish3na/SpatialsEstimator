# Spatials Estimator

*System Requirements :* CUDA 11.8, cudnn 8.7, Ubuntu >= 20.04, Python 3.10
*Sensor Requirements :* OAK-D-Pro Wide or OAK-D-Pro

To download segment-anything models [click here](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) (download vit_h: ViT-H SAM model) and place it in the following location

`segment-anything\segment_anything\models`

To install dependencies : `pip install -r requirements.txt`

To execute the code : `python3 run_spatials_estimator.py`

Please ensure the model paths of CREStereo and Segment-Anything are correct in "run_spatials_estimator.py" file before executing.

## Demo

![Demo Animation](spatials_estimator.gif)

[📹 Watch Full Demo Video (43MB)](spatials_estimator_demo.mp4)

## Gallery

### Process Flow
<img src="ProcessFlowChart.png" width="600">

### Input Images
<p align="center">
  <img src="RGB_Image.png" width="400" alt="RGB Image">
  <img src="Left_Stereo_Image.png" width="400" alt="Left Stereo Image">
</p>
<p align="center">
  <img src="Right_Stereo_Image.png" width="400" alt="Right Stereo Image">
</p>

### Segmentation Results
<p align="center">
  <img src="Segmented_Image.png" width="400" alt="Segmented Image">
  <img src="image_segmented.png" width="400" alt="Image Segmented">
</p>

### ROI Detection
<p align="center">
  <img src="image_roi.png" width="400" alt="ROI">
  <img src="image_roi2.png" width="400" alt="ROI 2">
</p>

### Depth Estimation
<p align="center">
  <img src="image_depth.png" width="400" alt="Depth Map">
  <img src="Depths_of_Filtered_Centroids.png" width="400" alt="Depths of Filtered Centroids">
</p>

### Spatial Estimation Results
<p align="center">
  <img src="image_centroid_spatials.png" width="400" alt="Centroid Spatials">
</p>
<p align="center">
  <img src="image_spatials2.png" width="400" alt="Spatials 2">
  <img src="image_spatials3.png" width="400" alt="Spatials 3">
</p>
<p align="center">
  <img src="image_spatials4.png" width="400" alt="Spatials 4">
</p>

### System Overview
<img src="Interactive%20Spatial%20Estimator.png" width="800" alt="Interactive Spatial Estimator">
