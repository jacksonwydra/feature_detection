import numpy as np
from scipy import signal
from scipy import ndimage
from typing import Tuple
from copy import deepcopy


def image_gradients(img_bw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
    sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)

    Ix = signal.convolve2d(img_bw, sobel_x_kernel, mode="same")
    Iy = signal.convolve2d(img_bw, sobel_y_kernel, mode="same")

    return Ix, Iy


def second_moments(
    Ix: np.ndarray, Iy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ixx = ndimage.gaussian_filter(Ix**2, 1)
    Iyy = ndimage.gaussian_filter(Iy**2, 1)
    Ixy = ndimage.gaussian_filter(Ix * Iy, 1)

    return Ixx, Iyy, Ixy


def harris_response_map(
    Ixx: np.ndarray, Iyy: np.ndarray, Ixy: np.ndarray, k=0.05
) -> np.ndarray:
    det = Ixx * Iyy - Ixy**2
    trace = Ixx + Iyy
    hrm = det - k * trace**2

    return hrm


def find_corners(hrm: np.ndarray) -> np.ndarray:
    corners = deepcopy(hrm)

    corners[corners <= 0] = 0
    mean = np.mean(corners[corners != 0])
    corners[corners <= mean] = 0
    corners[corners > mean] = 255

    return corners
