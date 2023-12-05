import torch
import numpy as np
import cv2

# define a bunch of augmentations (either take in rasterized image or vectorized image)

def get_random_curve(amp):
    freqs = np.random.uniform(0.64, 20, 10)
    amps = np.random.uniform(-1, 1, 10) * amp 
    return lambda x: np.sum([amp * np.sin(freq * x) for freq, amp in zip(freqs, amps)])

def vert_curve_image_raw(img, amp=0.05, fill=0):
    curve = get_random_curve(amp)
    for i in range(img.shape[2]):
        roll_val = int(torch.clamp(torch.tensor(curve(i / img.shape[2])), -0.2, 0.2) * img.shape[1])
        img[0, :, i] = torch.roll(img[0, :, i], roll_val)
        if roll_val < 0:
            img[0, roll_val:, i] = fill
        else:
            img[0, :roll_val, i] = fill
    return img

def hori_curve_image_raw(img, amp=0.05, fill=0):
    curve = get_random_curve(amp)
    for i in range(img.shape[1]):
        roll_val = int(torch.clamp(torch.tensor(curve(i / img.shape[1])), -0.2, 0.2) * img.shape[2])
        img[0, i, :] = torch.roll(img[0, i, :], roll_val)
        if roll_val < 0:
            img[0, i, roll_val:] = fill
        else:
            img[0, i, :roll_val] = fill
    return img


def point_line_distance(start, end, point):
    if not isinstance(start, np.ndarray):
        start = np.array(start)
    if not isinstance(end, np.ndarray):
        end = np.array(end)
    if not isinstance(point, np.ndarray):
        point = np.array(point)
    line_vec = end - start
    point_vec = point - start
    cross_product = np.cross(line_vec, point_vec)
    line_length = np.linalg.norm(line_vec)
    distance = np.abs(cross_product) / line_length
    return distance


def curve_line(start, end, line_thickness, img_shape, amp=0.05, fill=1.0):
    curvev = get_random_curve(amp)
    curveh = get_random_curve(amp)
    img = np.ones(img_shape, np.float32) #img = torch.from_numpy(np.zeros(img_shape, np.float32))
    img = cv2.line(img, start, end, 0.0, line_thickness)
    width = abs(end[0] - start[0] + 1)
    height = abs(end[1] - start[1] + 1)
    length = (width**2 + height**2 + 0.000001) ** 0.5
    thickness = 0
    for x in range(start[0], end[0]+1):
        roll_val = int(torch.clamp(torch.tensor(curvev(x / width)), -0.2, 0.2) * length)
        img[:, x] = torch.roll(torch.from_numpy(img[:, x]), roll_val)
        if roll_val < 0:
            img[roll_val:, x] = fill
        else:
            img[:roll_val, x] = fill

        y = (end[1] - start[1])/(end[0] - start[0] + 0.001) * (x - start[0]) + start[1] 
        print(x, y)
        y += roll_val
        deltax = int(torch.clamp(torch.tensor(curveh(y / height)), -0.2, 0.2) * length)
        print(deltax, roll_val)
        nx = x + deltax
        d = point_line_distance(start, end, np.array([nx, y]))
        print(d)
        thickness = max(thickness, 2*d)

    for y in range(start[1], end[1]+1):
        roll_val = int(torch.clamp(torch.tensor(curveh(y / height)), -0.2, 0.2) * length)
        img[y, :] = torch.roll(torch.from_numpy(img[y, :]), roll_val)
        if roll_val < 0:
            img[y, roll_val:] = fill
        else:
            img[y, :roll_val] = fill

    return img, thickness + line_thickness


