import numpy as np
from PIL import Image
import random
import torch
from PIL import ImageEnhance

class random_rot(object):
    def __init__(self, d):
        self.d = d
    def __call__(self, img):
        p = np.random.random()
        if p < 1.0:
            f = np.random.randint(-self.d,self.d)
            img = img.rotate(-f)
        return img

class random_resize(object):
    def __init__(self, min_value, max_value, min_size):
        self.min_value = min_value
        self.max_value = max_value
        self.min_size = min_size
    def __call__(self, img):
        min_size = self.min_size
        scale = np.random.uniform(self.min_value, self.max_value)
        size = img.size[0]

        size_new = int(scale * size)

        img = img.resize((size_new, size_new))

        if size_new < min_size:
            canvas = Image.new('RGB', (min_size,min_size), 0)
            x = random.randint(0, min_size - size_new)
            y = random.randint(0, min_size - size_new)
            canvas.paste(img, (x,y))
            img = canvas

        return img

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

jitter_param  = dict(Brightness=0.4, Contrast=0.4, Color=0.4)
transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)

class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]


    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out