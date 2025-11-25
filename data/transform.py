#!/usr/bin/python3
#coding=utf-8

from configparser import Interpolation
import cv2
from torchvision.transforms.functional import rotate, InterpolationMode
import torch
import numpy as np

class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, mask, P_fg, P_bg):
        for op in self.ops:
            image, mask, P_fg, P_bg = op(image, mask, P_fg, P_bg)
        return image, mask, P_fg, P_bg

class RGBDCompose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, depth, mask):
        for op in self.ops:
            image, depth, mask = op(image, depth, mask)
        return image, depth, mask


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, image, mask, P_fg, P_bg):
        image = (image - self.mean) / self.std
        return image, mask, P_fg, P_bg
    
class RGBDNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, image, depth, mask):
        image = (image - self.mean)/self.std
        depth = (depth - self.mean)/self.std
        mask /= 255
        return image, mask

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, P_fg, P_bg):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        P_fg  = cv2.resize(P_fg, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        P_bg  = cv2.resize(P_bg, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, P_fg, P_bg

class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, P_fg, P_bg):
        H, W, _ = image.shape
        xmin = np.random.randint(W - self.W + 1)
        ymin = np.random.randint(H - self.H + 1)
        image = image[ymin:ymin+self.H, xmin:xmin+self.W, :]
        mask  = mask[ymin:ymin+self.H, xmin:xmin+self.W, :]
        P_fg  = P_fg[ymin:ymin+self.H, xmin:xmin+self.W, :]
        P_bg  = P_bg[ymin:ymin+self.H, xmin:xmin+self.W, :]
        return image, mask, P_fg, P_bg

class RandomHorizontalFlip(object):
    def __call__(self, image, mask, P_fg, P_bg):
        if np.random.randint(2) == 1:
            image = image[:, ::-1, :].copy()
            mask  = mask[:, ::-1, :].copy()
            P_fg  = P_fg[:, ::-1, :].copy()
            P_bg  = P_bg[:, ::-1, :].copy()
        return image, mask, P_fg, P_bg

class ToTensor(object):
    def __call__(self, image, mask, P_fg, P_bg):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        mask  = mask.permute(2, 0, 1)
        P_fg  = torch.from_numpy(P_fg)
        P_fg  = P_fg.permute(2, 0, 1)
        P_bg  = torch.from_numpy(P_bg)
        P_bg  = P_bg.permute(2, 0, 1)
        return image, mask.mean(dim=0, keepdim=True), P_fg.mean(dim=0, keepdim=True), P_bg.mean(dim=0, keepdim=True)

class Flip:
    def __init__(self, flip_num):
        assert flip_num in [0,1,2]
        self.flip = flip_num
    def __call__(self, img, msk = None):
        if self.flip==1:
            img = img.flip(-2)
        elif self.flip==2:
            img = img.flip(-1)
        return img, msk

class Rotate:
    def __init__(self, rot_degree):
        self.rot = rot_degree
    def __call__(self, img, msk = None):
        img = rotate(img, self.rot, interpolation=InterpolationMode.BILINEAR)
        return img, msk

class RandomNoise:
    def __init__(self, noise_level):
        self.noise_level = noise_level
    def __call__(self, img, msk = None):
        noise = torch.randn_like(img) * self.noise_level
        img = img + noise
        return img, msk



