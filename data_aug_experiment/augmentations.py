import torch
import numpy as np

# define a bunch of augmentations (either take in rasterized image or vectorized image)

def get_random_curve(amp):
    freqs = np.random.uniform(0.64, 50, 10)
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
