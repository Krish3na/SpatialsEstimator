import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
from SpatialsEstimator import spatials_estimator as se

flag = True
spatials_estimator = se.SpatialsEstimator()

sam_model_path = "./segment-anything/segment_anything/models/sam_vit_h_4b8939.pth"
crestereo_model_path = f'./SpatialsEstimator/models/crestereo_combined_iter10_720x1280.onnx'

mask_generator, depth_estimator, device, camera_config = spatials_estimator.load_model(sam_model_path= sam_model_path, crestereo_model_path= crestereo_model_path)

roi_corners = list(spatials_estimator.find_roi().values())
print("\nROI Corners : ", roi_corners)

iter_num = 0

while flag:
    
    left_frame, rgb_frame, right_frame = spatials_estimator.get_output_frames()
    iter_num += 1
    #key = cv2.waitKey(1) & 0xFF 

    if iter_num > 10:
        # Depth Maps
        disparity_map, depth_map = spatials_estimator.get_maps(left_frame, right_frame)
        
        #(1)
        #if key == ord('r'):  # Capture and save RGB image ----- DB
        cv2.imwrite("RGB_Image.png", rgb_frame)
        print("RGB Image is saved")


        #(2)
        #if key == ord('c'):  # Capture and save Stereo images ---- DB
        cv2.imwrite("Left_Stereo_Image.png", left_frame)
        cv2.imwrite("Right_Stereo_Image.png", right_frame)
        print("Stereo Images are saved")

        #(3)
        #if key == ord('s') : # Run SAM and generate masks ---- DB
        roi_image = spatials_estimator.create_roi(left_frame, roi_corners)
        roi_image = roi_image.astype(np.uint8)
        spatials_estimator.generate_masks(roi_image)

        #plt.figure(figsize=(20,20))
        plt.get_current_fig_manager().set_window_title('Segmented Image')
        plt.imshow(roi_image)
        image_with_annotations = spatials_estimator.show_annotations()
        plt.imshow(image_with_annotations)
        plt.axis('off')
        plt.savefig("Segmented_Image.png")  # Save the image
        plt.show(block=False) 
        plt.pause(3)
        plt.gcf().canvas.flush_events()
        plt.close()

        masks_info = spatials_estimator.save_masked_images(roi_image)
        #print(masks_info)

        #(4)
        #if key == ord('b'): # Filter and create bounding boxes, and centroids ---- DB
        bounded_masks_info, centroid_coordinates = spatials_estimator.draw_bounding_boxes_and_find_centroids(roi_corners) #, masks_info)
        # print(centroid_coordinates)
        # for i in list(bounded_masks_info.values()):
        #     print(i[1])

        #(5)
        #if key == ord('d'): # Estimate spatial coordinates for provided centroids ----- DB
        # Estimate depth for centroids and display depth image
        spatial_coordinates = spatials_estimator.get_spatial_coordinates() #, centroid_coordinates)
        print("\nSpatial Coordinates are : ", spatial_coordinates)
        print("\nSpatial Coordinates dictionary length is : ", len(spatial_coordinates))

        filtered_spatial_coordinates = spatials_estimator.filter_spatial_coordinates() #, spatial_coordinates)
        print("\nFinal Spatial Coordinates length is : ", len(filtered_spatial_coordinates))
        print("\nFinal Spatial Coordinates are : ", filtered_spatial_coordinates)

        spatials_estimator.plot_spatial_coordinates() #, filtered_spatial_coordinates, centroid_coordinates)

        shutil.rmtree("bounded_output")
        shutil.rmtree("output")

    # if key == ord('q'):
    #     exit()

    if iter_num == 11:
        flag = False
        
            
