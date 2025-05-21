# Spatials Estimator

*System Requirements :* CUDA 11.8, cudnn 8.7, Ubuntu >= 20.04, Python 3.10
*Sensor Requirements :* OAK-D-Pro Wide or OAK-D-Pro

To download segment-anything models [click here](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) (download vit_h: ViT-H SAM model) and place it in the following location

`segment-anything\segment_anything\models`

To install dependencies : `pip install -r requirements.txt`

To execute the code : `python3 run_spatials_estimator.py`

Please ensure the model paths of CREStereo and Segment-Anything are correct in "run_spatials_estimator.py" file before executing.

## Demo

[📹 Watch Demo Video (43MB)](spatials_estimator_demo.mp4)

![Demo Thumbnail](RGB_Image.png)

## Gallery

### Process Flow
![Process Flow Chart](ProcessFlowChart.png)

### Input Images
![RGB Image](RGB_Image.png)
![Left Stereo Image](Left_Stereo_Image.png)
![Right Stereo Image](Right_Stereo_Image.png)

### Segmentation Results
![Segmented Image](Segmented_Image.png)
![Image Segmented](image_segmented.png)

### ROI Detection
![ROI](image_roi.png)
![ROI 2](image_roi2.png)

### Depth Estimation
![Depth Map](image_depth.png)
![Depths of Filtered Centroids](Depths_of_Filtered_Centroids.png)

### Spatial Estimation Results
![Centroid Spatials](image_centroid_spatials.png)
![Spatials 1](image_spatials1.png)
![Spatials 2](image_spatials2.png)
![Spatials 3](image_spatials3.png)
![Spatials 4](image_spatials4.png)

### System Overview
![Interactive Spatial Estimator](Interactive%20Spatial%20Estimator.png)
