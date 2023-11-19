import numpy as np

# define a bunch of augmentations (either take in rasterized image or vectorized image)

def get_random_curve():
    freqs = np.random.uniform(0.1, 1, 10)
    amps = np.random.uniform(-1, 1, 10)
    return lambda x: np.sum([amp * np.sin(freq * x) for freq, amp in zip(freqs, amps)])

def curve_image_raw(img):
    curve = get_random_curve()
    for i in range(img.shape[1]):
        roll_val = int(curve(i) * img.shape[0])
        img[:, i] = np.roll(img[:, i], roll_val)
        if roll_val < 0:
            img[roll_val:, i] = 0
        else:
            img[:roll_val, i] = 0
    return img