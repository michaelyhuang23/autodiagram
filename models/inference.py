import numpy as np
import cv2

def inference_line(model, img):
    rets = model(img)
    start = rets[:2]
    end = rets[2:4]
    thickness = rets[4]
    img_dim = (img.shape[0]**2  + img.shape[1]**2)**0.5
    if np.linalg.norm(end - start) < 0.05 * img_dim:
        return None, None
    line_img = np.zeros(img.shape, dtype=np.float32)
    line_img = cv2.line(line_img, start, end, 1.0, thickness + 2)
    return (start, end, thickness), np.clip(img + line_img, 0, 1)


def inference(model, img):
    total_tikz = []
    for i in range(50):
        line, img = inference_line(model, img)
        if line is None:
            break
        start, end, thickness = line
        line_tikz = f'\draw ({start[0]},{start[1]}) -- ({end[0]},{end[1]});'
        total_tikz.append(line_tikz)
    return '\n'.join(total_tikz)





