import numpy as np
from skimage.transform import hough_line
from skimage.filters import sobel_h, gaussian
from skimage.transform import hough_line, hough_line_peaks, resize
from skimage.filters import sobel_h, gaussian
from skimage.feature import canny

class ToBand(object):
    """
        Extracts the band from the albedo

        Args:
            height: height of the output image
    """
    def __init__(self, height, theta_res=180, margin=5):
        self.height = height
        self.theta_res = theta_res
        self.margin = margin

    def __call__(self, img):
        """
        Applies the transformation

        :param img: the input image in format ndarray
        :return: an image in format ndarray containing the albedo band, the size will be self.height x width
        here "width" is the original image width
        """
        dist = self.detect_parallel_cut(img)
        img = img[0:dist+self.margin, :]
        img = resize(img, (self.height, img.shape[1]))
        img.shape = (img.shape[0], img.shape[1], 1)
        return img

    def detect_parallel_cut(self, image):
        edges = canny(image, sigma=2).astype(np.float32)
        out, theta, d = hough_line(edges, theta=np.linspace(0, np.pi, self.theta_res))
        _, angles, distances = hough_line_peaks(out, theta, d, threshold=0.35*np.max(out))
        idx_parallel = np.logical_or(angles > np.pi / 2 - 2 * np.pi / self.theta_res,
                                     angles < np.pi / 2 + 2 * np.pi / self.theta_res)
        distances = distances[idx_parallel]
        max_dist = np.max(distances)
        max_dist = np.round(max_dist).astype(int)
        return max_dist


class Crop(object):
    """
        Extracts the band from the albedo

        Args:
            height: height of the output image
    """
    def __init__(self, height):
        self.height = height

    def __call__(self, img):
        img = img[0:self.height, :]
        img.shape = (img.shape[0], img.shape[1], 1)
        return img
