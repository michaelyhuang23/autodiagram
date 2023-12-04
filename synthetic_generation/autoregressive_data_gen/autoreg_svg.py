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

import cv2, random, sys
import numpy as np

sys.path.append('../../data_aug_experiment')
from augmentations import curve_line

# Create a blank image
def create_blank_image(width, height, color=(255, 255, 255)):
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = color
    return image

# Draw a line on the image
def draw_line(image, start_point, end_point, line_color=(0, 0, 0), line_thickness=2):
    cv2.line(image, start_point, end_point, line_color, line_thickness)


# Image dimensions and line parameters
width, height = 224, 224

num_data_points = 3
for i in range(num_data_points):
    # Create a blank image
    image = create_blank_image(width, height)

    num_lines_to_draw = random.randint(1,5)

    for j in range(num_lines_to_draw):
        # Draw a line by defining the two endpoints. 
        start_point = (random.randint(0, width), random.randint(0, height))
        end_point = (random.randint(0, width), random.randint(0, height))
    
    # Sort the lines based on length:
        
    for j in range(num_data_points):
        line_color = (0, 0, 0)
        line_thickness = 2

        draw_line(image, start_point, end_point, line_color, line_thickness)

        # Save the image
        output_filename = f'imgfiles/line_drawing_test_{i}-{j}.jpg'
        cv2.imwrite(output_filename, image)        

#todo: 
# - make the lines increase in size
# - figure out how to pass to michael's code
# - set up the folder for saving the images

