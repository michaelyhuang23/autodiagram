import cv2
import numpy as np

class VectorObject:
    def __init__(self, category, values):
        if category not in ["rectangle", "circle", "line"]:
            raise Exception("Category not supported")
        self.category = category
        self.values = values
    
    def render(self, img):
        if self.category == "rectangle":
            cv2.rectangle(img, (self.values['x1'], self.values['y1']), (self.values['x2'], self.values['y2']), 255, self.values['thickness'])
        elif self.category == "circle":
            cv2.circle(img, (self.values['x'], self.values['y']), self.values['radius'], 255, self.values['thickness'])
        elif self.category == "line":
            cv2.line(img, (self.values['x1'], self.values['y1']), (self.values['x2'], self.values['y2']), 255, self.values['thickness'])
        else:
            raise ValueError("Category not supported")
    
    def min_coords(self):
        if self.category == "rectangle":
            return (self.values['x1'], self.values['y1'])
        elif self.category == "circle":
            return (self.values['x'] - self.values['radius'], self.values['y'] - self.values['radius'])
        elif self.category == "line":
            return (min(self.values['x1'], self.values['x2']), min(self.values['y1'], self.values['y2']))
        else:
            raise ValueError("Category not supported")
    
    def max_coords(self):
        if self.category == "rectangle":
            return (self.values['x2'], self.values['y2'])
        elif self.category == "circle":
            return (self.values['x'] + self.values['radius'], self.values['y'] + self.values['radius'])
        elif self.category == "line":
            return (max(self.values['x1'], self.values['x2']), max(self.values['y1'], self.values['y2']))
        else:
            raise ValueError("Category not supported")

class VectorImage:
    def __init__(self):
        self.objs = []
        self.canvas = (0, 0, 0, 0)
    
    def update_canvas(self, obj: VectorObject):
        self.canvas = (min(self.canvas[0], obj.min_coords()[0]), min(self.canvas[1], obj.min_coords()[1]), max(self.canvas[2], obj.max_coords()[0]), max(self.canvas[3], obj.max_coords()[1]))

    def add_obj(self, obj: VectorObject):
        self.update_canvas(obj)
        self.objs.append(obj)
    
    def render(self, img=None):
        if img is None:
            img = np.zeros((self.canvas[3] - self.canvas[1], self.canvas[2] - self.canvas[0], 1))
        for obj in self.objs:
            obj.render(img)
        return img