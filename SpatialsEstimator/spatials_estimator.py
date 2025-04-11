import os
import cv2
import math
import shutil
import numpy as np
import depthai as dai
from . import utility as u
from . import calc
from . import crestereo
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class SpatialsEstimator:
    def __init__(self):
        self.centroid_coordinates = {}
        self.spatial_coordinates = {}
        self.filtered_spatial_coordinates = {}
        self.cursor_position = (0, 0)
        self.points_dictionary = {}
        self.point_counter = 1
        self.masks_info = {}
        self.bounded_masks_info = {}
        print("Spatials Estimator object intitialized ...")

    def load_model(self, sam_model_path = "", crestereo_model_path = "", sam_model_type = "vit_h", sam_machine = "cuda", crestereo_max_distance = 3, crestereo_focal_length_in_pixels = -1):

        if len(sam_model_path):
            sam = sam_model_registry[sam_model_type](checkpoint=sam_model_path)
            sam.to(device=sam_machine)
            self.mask_generator = SamAutomaticMaskGenerator(sam)
            print('SAM Model loaded')
        else:
            print("Please ensure the SAM model path is correct !")

        if len(crestereo_model_path):
            self.pipeline = self.create_pipeline()
            self.device = dai.Device(self.pipeline)
            calibData = self.device.readCalibration()
            intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C)
            crestereo_focal_length_in_pixels = intrinsics[0][0]

            #Initializing Camera Config
            self.camera_config = crestereo.CameraConfig(0.075, crestereo_focal_length_in_pixels)#0.5*input_shape[1]/0.72) # 71.9 deg. FOV 
            # Initialize model object
            self.depth_estimator = crestereo.CREStereo(crestereo_model_path, camera_config= self.camera_config, max_dist= crestereo_max_distance)
            print('CREStereo Model loaded')
        else:
            print("Please ensure the CREStereo model path is correct !")

        return self.mask_generator, self.depth_estimator, self.device, self.camera_config
    
    def create_pipeline(self):
        # Create pipeline
        self.pipeline = dai.Pipeline()

        # Define sources and outputs
        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        monoRight = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        cam = self.pipeline.create(dai.node.ColorCamera)

        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam.setPreviewSize(1280,720)

        rect_left = self.pipeline.create(dai.node.XLinkOut)
        rect_right = self.pipeline.create(dai.node.XLinkOut)
        xout = self.pipeline.create(dai.node.XLinkOut)

        rect_left.setStreamName("rect_left")
        rect_right.setStreamName("rect_right")
        xout.setStreamName("rgb")

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # StereoDepth
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        stereo.rectifiedLeft.link(rect_left.input)
        stereo.rectifiedRight.link(rect_right.input)
        cam.preview.link(xout.input)

        return self.pipeline

    def get_output_frames(self):

        if self.device:
            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            self.rectified_left_queue = self.device.getOutputQueue(name="rect_left", maxSize=4, blocking=False)
            self.rectified_rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            self.rectified_right_queue = self.device.getOutputQueue(name="rect_right", maxSize=4, blocking=False)

            self.rectified_left_frame = self.rectified_left_queue.get().getCvFrame()
            self.rectified_rgb_frame = self.rectified_rgb_queue.get().getCvFrame()
            self.rectified_right_frame = self.rectified_right_queue.get().getCvFrame()

            self.rectified_left_frame = cv2.cvtColor(self.rectified_left_frame, cv2.COLOR_GRAY2BGR)
            self.rectified_right_frame = cv2.cvtColor(self.rectified_right_frame, cv2.COLOR_GRAY2BGR)

            return self.rectified_left_frame, self.rectified_rgb_frame, self.rectified_right_frame
    
    def get_maps(self, left_rectified_image, right_rectified_image) :
        # Depth Maps
        if len(left_rectified_image) and len(right_rectified_image) :
            self.disparity_map = self.depth_estimator(left_rectified_image, right_rectified_image)  # Disparity Map used to calculate the depth
            self.depth_map = self.depth_estimator.get_depth_from_disparity(self.disparity_map, self.camera_config) # 2D Depth Map for calculating spacitial coordinates

            return self.disparity_map, self.depth_map
    
    def on_mouse_event(self, event, x, y, flags, param):

        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor_position = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Save the point with a key like 'point_1', 'point_2', ...
            self.point_key = f'point_{self.point_counter}'
            self.points_dictionary[self.point_key] = [x, y]
            self.point_counter += 1

    def draw_points_and_lines(self, frame):
        self.frame_with_points_and_lines = frame.copy()

        # Draw cursor
        cv2.line(self.frame_with_points_and_lines, (self.cursor_position[0], 0), (self.cursor_position[0], frame.shape[0]), (0, 0, 255), 1)
        cv2.line(self.frame_with_points_and_lines, (0, self.cursor_position[1]), (frame.shape[1], self.cursor_position[1]), (0, 0, 255), 1)

        # Draw x, y values next to the cursor
        cv2.putText(self.frame_with_points_and_lines, f'({self.cursor_position[0]}, {self.cursor_position[1]})', (self.cursor_position[0] + 5, self.cursor_position[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw points
        first_point_drawn = False  # Track whether the first point has been drawn
        for key, point in self.points_dictionary.items():
            color = (0, 255, 0) if not first_point_drawn else (255, 0, 0)
            
            # Check if it's the last point and there are at least 4 points
            if len(self.points_dictionary) >= 4 and point == list(self.points_dictionary.values())[-1]:
                # Draw a line connecting the last point to the first point
                first_point = list(self.points_dictionary.values())[0]
                cv2.line(self.frame_with_points_and_lines, (point[0], point[1]), (first_point[0], first_point[1]), (0, 255, 0), 2)

            cv2.circle(self.frame_with_points_and_lines, point, 2, color, -1)
            # Draw x, y values next to each point
            cv2.putText(self.frame_with_points_and_lines, f'{key[6:]}: ({point[0]}, {point[1]})', (point[0] + 5, point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if key == 'point_1':  # Check if the first point is being drawn
                first_point_drawn = True

        # Draw lines
        points = list(self.points_dictionary.values())
        for i in range(1, len(points)):
            cv2.line(self.frame_with_points_and_lines, points[i - 1], points[i], (0, 255, 0), 2)

        return self.frame_with_points_and_lines
    
    def find_roi(self):
        print("\nClick the points on the Streaming Window to create an ROI ... \nOptions: After marking the points \n Press 'p' to continue with execution\n Press 'e' to erase points and redraw \n Press 'm' to see the spatial information with free moving cursor \n \n If pressed 'm' please click to enable the cursor and click again to disable it ...\n")

        # Create a window and set the callback function for mouse events
        cv2.namedWindow('Streaming Window')
        cv2.setMouseCallback('Streaming Window', self.on_mouse_event)

        while True:

            left_frame, _, _ = self.get_output_frames()

            # Draw points and lines
            self.frame_with_points_and_lines = self.draw_points_and_lines(left_frame)

            # Show the frame with points and lines
            cv2.imshow('Streaming Window', self.frame_with_points_and_lines)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('e'):
                # Erase points and lines
                self.points_dictionary = {}
                self.point_counter = 1

            elif key == ord('p'):
                # Print the points
                cv2.destroyAllWindows()
                return self.points_dictionary
            
            elif key == ord('m'):
                # Print the points
                cv2.destroyWindow('Streaming Window')
                cv2.destroyAllWindows()
                self.see_spatial_coordinates_moving_cursor()
            
            elif key == ord('q'):
                exit()
    
    def create_roi(self, frame, roi_corners):
        # Create a mask
        mask = np.zeros_like(frame)

        # Convert points to NumPy array
        points = np.array(roi_corners, np.int32)
        points = points.reshape((-1, 1, 2))

        # Fill the region inside the polygon defined by the points with white
        cv2.fillPoly(mask, [points], (255, 255, 255))

        # Apply the mask to the image
        roi_image = cv2.bitwise_and(frame, mask)

        return roi_image
    
    def generate_masks(self, image) :

        if len(image):
            self.masks = self.mask_generator.generate(image)
            print('Masks generated')
            return self.masks
        
    def show_annotations(self):
        if len(self.masks) == 0:
            return
        sorted_annotations = sorted(self.masks, key=(lambda x: x['area']), reverse=True)
        #ax = plt.gca()
        #ax.set_autoscale_on(False)

        segmented_image = np.ones((sorted_annotations[0]['segmentation'].shape[0], sorted_annotations[0]['segmentation'].shape[1], 4))
        segmented_image[:,:,3] = 0
        for annotation in sorted_annotations:
            mask = annotation['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            segmented_image[mask] = color_mask
            
        #ax.imshow(segmented_image)
        return segmented_image
    
    def save_masked_images(self, image, output_path='output'):
        # Create the output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        for idx, mask in enumerate(self.masks):
            mask_binary = mask['segmentation'].astype(np.uint8)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a mask image with the same size as the input image
            mask_img = np.zeros_like(image)

            # Draw the contours on the mask image
            cv2.drawContours(mask_img, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

            # Bitwise AND operation to get the masked image
            masked_image = cv2.bitwise_and(image, mask_img)

            # Save the cropped image
            cv2.imwrite(os.path.join(output_path, f'masked_image_{idx}.png'), cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
            
            # Initialize dictionary entry
            self.masks_info[f'masked_image_{idx}.png'] = []

            # Append values to the dictionary entry
            self.masks_info[f'masked_image_{idx}.png'] += [cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR), mask['area'], mask['crop_box'], mask['bbox']]

        print(f"Masks saved in '{output_path}' folder")

        return self.masks_info
    
    def calculate_roi_area(self, roi_corners):
        # Make sure there are exactly four points
        if len(roi_corners) == 4:
            # Add the first point at the end to close the polygon
            roi_corners = roi_corners + [roi_corners[0]]

            # Apply the shoelace formula
            area = 0.5 * abs(sum(x0*y1 - x1*y0 for (x0, y0), (x1, y1) in zip(roi_corners, roi_corners[1:])))

            return area
        else:
            print("ROI should have exactly four points.")
            return None
        
    def draw_bounding_boxes_and_find_centroids_processing(self, masked_image, mask_name, min_area, max_area, output_folder = 'bounded_output'):

        # Find contours
        contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if there is a single contour
        if len(contours) == 1:
            # Process the single contour
            cntr = contours[0]

            # Check if the contour has approximately four corner points
            perimeter = cv2.arcLength(cntr, True)
            approx = cv2.approxPolyDP(cntr, 0.05 * perimeter, True)

            if len(approx) == 4:
                # Create result image
                result_image = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR)

                # Draw rotated bounding box
                rect = cv2.minAreaRect(cntr)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # Find and print the area of the contour
                cntr_area = cv2.contourArea(cntr)
                #print(f"Contour area: {roi_area, cntr_area}")

                if min_area < cntr_area < max_area :

                    # Calculate aspect ratio of bounding rectangle
                    width, height = rect[1]
                    aspect_ratio = width / height if height != 0 else 0

                    # Check aspect ratio to filter out irregular shapes
                    if 0.5 <= aspect_ratio <= 2.0:  # Adjust the range as needed
                        # Draw the bounding box
                        cv2.drawContours(result_image, [box], 0, (0, 255, 0), 1)

                        # Draw centroid
                        M = cv2.moments(cntr)
                        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
                        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
                        cv2.circle(result_image, (cx, cy), 1, (0, 0, 255), -1)

                        # Store centroid coordinates
                        self.centroid_coordinates[mask_name] = [cx, cy]

                        # Save the result image
                        filename = os.path.join(output_folder, mask_name)
                        # Initialize dictionary entry
                        self.bounded_masks_info[mask_name] = []

                        # Append values to the dictionary entry
                        self.bounded_masks_info[mask_name] += [result_image, cntr_area]

                        cv2.imwrite(filename, result_image)
                    else:
                        print(f"Ignore irregular shape: {mask_name} (Aspect Ratio: {aspect_ratio})")

                else:
                    print(f"Ignore contour as it is ROI boundaries: {mask_name}")

            else:
                print(f"Ignore contour without approximately four corner points: {mask_name}")
        else:
            print(f"Ignore image: {mask_name} as it does not have a single contour.")

    def draw_bounding_boxes_and_find_centroids(self, roi_corners, masks_info = None, output_folder = 'bounded_output'):
        if masks_info is None:
            masks_info = self.masks_info
        
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        roi_area = self.calculate_roi_area(roi_corners)
        min_area = 3000
        max_area = roi_area - 2000

        # List all files in the input folder
        for mask_name, mask_info in masks_info.items():
            if len(mask_info):  
                masked_image = cv2.cvtColor(mask_info[0], cv2.COLOR_BGR2GRAY)
                # Apply bounding box drawing and centroid marking
                self.draw_bounding_boxes_and_find_centroids_processing(masked_image, mask_name, min_area, max_area)

        return self.bounded_masks_info, self.centroid_coordinates
    
    def get_spatial_coordinates(self, centroid_coordinates = None):
        if centroid_coordinates is None:
            centroid_coordinates = self.centroid_coordinates

        text = u.TextHelper()
        hostSpatials = calc.HostSpatialsCalc(self.device)
        step = 6
        delta = 1
        hostSpatials.setDeltaRoi(delta)

        # Normalize and convert depth map to CV_8U
        normalized_depth_map = cv2.normalize(self.depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_uint8 = cv2.convertScaleAbs(normalized_depth_map)

        # Calculate the z-coordinate (depth) in meters using the camera parameters
        z_color = cv2.applyColorMap(depth_map_uint8, cv2.COLORMAP_TURBO)

        for k, v in centroid_coordinates.items():
            x, y = v[0], v[1]
            spatials, centroid = hostSpatials.calc_spatials(self.depth_map, (x,y))
            spatials['x'] = spatials['x']*0.001
            spatials['y'] = spatials['y']*0.001
            spatials['z'] = spatials['z']*0.001

            self.spatial_coordinates[k] = []
            self.spatial_coordinates[k] += (spatials['x'], spatials['y'], spatials['z'])

        return self.spatial_coordinates
    
    def filter_spatial_coordinates(self, spatial_coordinates = None, min_depth = 1000000000000, e_min_depth = 0.0254):
        if spatial_coordinates is None:
            spatial_coordinates = self.spatial_coordinates
        
        for i in spatial_coordinates.values():
            if i[2] <= min_depth:
                min_depth = i[2]
                
        print("\nBlock with Min Depth is : ", min_depth)
            
        for k,v in spatial_coordinates.items():
            if (min_depth + e_min_depth)  >= v[2] >= (min_depth - e_min_depth): 
                self.filtered_spatial_coordinates[k] = [round(i, 4) for i in v]
                
        return self.filtered_spatial_coordinates
    
    def plot_spatial_coordinates(self, filtered_spatial_coordinates = None, centroid_coordinates = None):
        if filtered_spatial_coordinates is None:
            filtered_spatial_coordinates = self.filtered_spatial_coordinates

        if centroid_coordinates is None:
            centroid_coordinates = self.centroid_coordinates

        text = u.TextHelper()
        hostSpatials = calc.HostSpatialsCalc(self.device)
        step = 6
        delta = 1
        hostSpatials.setDeltaRoi(delta)

        # Normalize and convert depth map to CV_8U
        normalized_depth_map = cv2.normalize(self.depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_uint8 = cv2.convertScaleAbs(normalized_depth_map)

        # Calculate the z-coordinate (depth) in meters using the camera parameters
        z_color = cv2.applyColorMap(depth_map_uint8, cv2.COLORMAP_TURBO)

        for k, v in filtered_spatial_coordinates.items():
            x, y = centroid_coordinates[k]
            text.rectangle(z_color, (x-delta, y-delta), (x+delta, y+delta))
            text.putText(z_color, "X: " + ("{:.4f}m".format(v[0]) if not math.isnan(v[0]) else "--"), (x + 10, y + 20))
            text.putText(z_color, "Y: " + ("{:.4f}m".format(v[1]) if not math.isnan(v[1]) else "--"), (x + 10, y + 35))
            text.putText(z_color, "Z: " + ("{:.4f}m".format(v[2]) if not math.isnan(v[2]) else "--"), (x + 10, y + 50))

        cv2.imshow("Depths of Filtered Centroids", z_color)
        # Wait for 5000 milliseconds (5 seconds)
        cv2.waitKey(5000)
        cv2.imwrite("Depths_of_Filtered_Centroids.png", z_color)
        cv2.destroyAllWindows()

    def see_spatial_coordinates_moving_cursor(self):
        text = u.TextHelper()
        hostSpatials = calc.HostSpatialsCalc(self.device)
        step = 6
        delta = 1
        iter_num = 0
        hostSpatials.setDeltaRoi(delta)

        cursor_enabled = False
        cursor_position = (0, 0)

        def mouse_callback(event, x, y, flags, param):
            nonlocal cursor_enabled, cursor_position

            if event == cv2.EVENT_LBUTTONDOWN:
                cursor_enabled = not cursor_enabled
                cursor_position = (x, y)

            if cursor_enabled:
                # Update cursor_position continuously while the left mouse button is held down
                cursor_position = (x, y)

        print("Loading... Please Wait ...")

        cv2.namedWindow("Spatials Estimator")
        cv2.setMouseCallback("Spatials Estimator", mouse_callback)

        while True:
            iter_num += 1
            left_frame, _, right_frame = self.get_output_frames()

            if iter_num > 10:
                disparity_map, depth_map = self.get_maps(left_frame, right_frame)
                # Normalize and convert depth map to CV_8U
                normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                depth_map_uint8 = cv2.convertScaleAbs(normalized_depth_map)

                # Calculate the z-coordinate (depth) in meters using the camera parameters
                z_color = cv2.applyColorMap(depth_map_uint8, cv2.COLORMAP_JET)
                
                if cursor_enabled:
                    x, y = cursor_position
                    spatials, centroid = hostSpatials.calc_spatials(depth_map, (x, y))
                    spatials['x'], spatials['y'], spatials['z'] = spatials['x'] * 0.001, spatials['y'] * 0.001, spatials['z'] * 0.001

                    text.rectangle(z_color, (x - delta, y - delta), (x + delta, y + delta))
                    text.putText(z_color, "X: " + ("{:.4f}m".format(spatials['x']) if not math.isnan(spatials['x']) else "--"), (x + 10, y + 20))
                    text.putText(z_color, "Y: " + ("{:.4f}m".format(spatials['y']) if not math.isnan(spatials['y']) else "--"), (x + 10, y + 35))
                    text.putText(z_color, "Z: " + ("{:.4f}m".format(spatials['z']) if not math.isnan(spatials['z']) else "--"), (x + 10, y + 50))

                cv2.imshow("Spatials Estimator", z_color)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                exit()

                
