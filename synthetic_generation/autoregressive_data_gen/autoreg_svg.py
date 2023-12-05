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

# Draw a line on the image
def draw_line(image, start_point, end_point, line_color=(0, 0, 0), line_thickness=2):
    cv2.line(image, start_point, end_point, line_color, line_thickness)

def sort_point_pairs_by_distance(point_pairs):
    def distance(pair):
        (x0, y0), (x1, y1) = pair
        return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)

    return sorted(point_pairs, key=distance)

def generate_points(num_lines, num_rect=0, num_tri=0):
    points = []

    num_actual_lines = 0
    num_rectangles = 2
    num_actual_lines += 4 * num_rectangles
    for j in range(num_rectangles):
        print("rectangle")
        start_point = (random.randint(0, width-1), random.randint(0, height-1))
        end_point = (random.randint(0, width-1), random.randint(0, height-1))

        projection1 = (start_point[0], end_point[1])
        projection2 = (end_point[0], start_point[1])
        
        points.append([start_point, projection1])
        points.append([start_point, projection2])
        points.append([end_point, projection1])
        points.append([end_point, projection2])


    for j in range(num_lines_to_draw):
        # Draw a line by defining the two endpoints. 
        start_point = (random.randint(0, width-1), random.randint(0, height-1))
        end_point = (random.randint(0, width-1), random.randint(0, height-1))
        points.append([start_point, end_point])
    
    num_actual_lines += num_lines_to_draw
    

    # Sort the lines based on length:
    points = sort_point_pairs_by_distance(points)
    return points, num_actual_lines


# Image dimensions and line parameters
width, height = 224, 224

with open("line_coords_training.txt", "w") as f:
    num_data_points = 1
    for i in range(num_data_points):
        #Generate the points
        num_lines_to_draw = random.randint(1,3)
        points,num_lines = generate_points(num_lines_to_draw)

        # Create a blank image
        image = np.ones((width, height), np.float32)

        # Iteratively add the necessary lines
        for j in range(num_lines):
            line_thickness = 2

            # Call data augmentation function which returns curved line
            additive_img, variance_width = curve_line(points[j][0], points[j][1], line_thickness, (224,224), amp=0.01)
            variance_width = round(variance_width,1)
            points[j].append(variance_width)

            # draw_line(image, start_point, end_point, line_color, line_thickness)
            image = np.clip(image * additive_img,0,1)

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
# - (done) add rectangle
# - add triangle
# make rectangle straighter?

