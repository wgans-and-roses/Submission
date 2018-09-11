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
