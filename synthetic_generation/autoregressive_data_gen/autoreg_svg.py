# Need to pass on:
# i-j images
# coordinates of lines
# thickness of lines
# rasterize everything at end

# What I need to do
# - for every single image sequence, generate the entire package:
# - begin blank canvas
# - num_lines = random number from 1 to 10
# - for i in range(num_lines):
#     - opencv draw the line onto an addition.svg
#     - get a rasterized version of the line/addition.svg
#     - save the coordinates of the line
#     - save a version of the rasterized svg file

import cv2, random, sys, math, torch
import numpy as np

sys.path.append('../../data_aug_experiment')
from augmentations import curve_line

# # Create a blank image
# def create_blank_image(width, height, color=(255, 255, 255)):
#     image = np.zeros((height, width, 3), np.uint8)
#     image[:] = color
#     return image

# Draw a line on the image
def draw_line(image, start_point, end_point, line_color=(0, 0, 0), line_thickness=2):
    cv2.line(image, start_point, end_point, line_color, line_thickness)

def sort_point_pairs_by_distance(point_pairs):
    def distance(pair):
        (x0, y0), (x1, y1) = pair
        return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)

    return sorted(point_pairs, key=distance)

# Image dimensions and line parameters
width, height = 224, 224

with open("line_coords_training.txt", "w") as f:
    num_data_points = 5
    for i in range(num_data_points):
        num_lines_to_draw = random.randint(1,5)
        points = []

        for j in range(num_lines_to_draw):
            # Draw a line by defining the two endpoints. 
            start_point = (random.randint(0, width), random.randint(0, height))
            end_point = (random.randint(0, width), random.randint(0, height))
            points.append([start_point, end_point])
        
        # Sort the lines based on length:
        points = sort_point_pairs_by_distance(points)
        print(points)

        # Create a blank image
        image = np.ones((width, height), np.float32)

        # Iteratively add the necessary lines
        for j in range(num_lines_to_draw):
            line_color = (0, 0, 0)
            line_thickness = 2

            # Call data augmentation function which returns curved line
            additive_img, variance_width = curve_line(points[j][0], points[j][1], line_thickness, (224,224), amp=0.01)
            variance_width = round(variance_width,1)
            # Record the thickness
            points[j].append(variance_width)

            print("image and then additive image:")
            print(image)
            print(additive_img)
            image = np.clip(image * additive_img,0,1)
            # draw_line(image, start_point, end_point, line_color, line_thickness)

            # Save the image
            output_filename = f'imgfiles/line_drawing_test_{i}-{j}.jpg'
            cv2.imwrite(output_filename, np.round(image*255))
        
        
        # Write the idealized data onto the text file
        for point in reversed(points):
            f.write(str(point))
        f.write("\n")

#todo: 
# - (done) make the lines increase in size
# - (done) figure out how to call michael's data aug
# - (done) set up the folder for saving the images
# - add polygons

