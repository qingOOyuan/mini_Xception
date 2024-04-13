import numpy as np
import cv2
import random

# Define a class for augmenting images with a list of transformations
class ImageAugmenter:
    def __init__(self, transformations: list = None):
        if transformations is None:
            transformations = []
        self.transformations = transformations

    def __call__(self, image: np.ndarray) -> np.ndarray:
        for transformation in self.transformations:
            image = transformation(image)
        return image

# Define a class for adding salt and pepper noise to an image
class SaltAndPepperNoise:
    def __init__(self, noise_probability):
        self.noise_probability = noise_probability

    def __call__(self, image: np.ndarray):
        """
        Add salt and pepper noise to the image.
        :param image: Input image.
        :param noise_probability: Noise probability.
        :return: Image with added salt and pepper noise.
        """
        threshold = 1 - self.noise_probability
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                random_value = np.random.rand()
                if random_value < self.noise_probability:
                    image[i, j] = 0
                elif random_value > threshold:
                    image[i, j] = 255
        return image

# Define a class to apply a random horizontal shift to an image
class RandomHorizontalShift:
    def __init__(self, max_shift_rate):
        self.max_shift_rate = max_shift_rate

    def __call__(self, image):
        shift_rate = np.random.uniform(0, self.max_shift_rate)
        height, width = image.shape[:2]
        x_shift = int(width * shift_rate)
        if np.random.rand() < 0.5:
            x_shift = -x_shift
        transformation_matrix = np.float32([[1, 0, x_shift], [0, 1, 0]])
        shifted_image = cv2.warpAffine(image, transformation_matrix, (width, height))
        return shifted_image

# Define a class to add Gaussian noise to an image
class GaussianNoise:
    def __init__(self, mean, standard_deviation, percentage):
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.percentage = percentage

    def __call__(self, source_image):
        noisy_image = source_image
        noise_pixel_count = int(self.percentage * source_image.size)
        for _ in range(noise_pixel_count):
            x_coord = random.randint(0, source_image.shape[0] - 1)
            y_coord = random.randint(0, source_image.shape[1] - 1)
            noisy_image[x_coord, y_coord] = (
                noisy_image[x_coord, y_coord] + random.gauss(self.mean, self.standard_deviation)
            )
            noisy_image[x_coord, y_coord] = np.clip(noisy_image[x_coord, y_coord], 0, 255)
        return noisy_image

# Define a class to apply a random vertical shift to an image
class RandomVerticalShift:
    def __init__(self, max_shift_rate):
        self.max_shift_rate = max_shift_rate

    def __call__(self, image):
        shift_rate = np.random.uniform(0, self.max_shift_rate)
        height, width = image.shape[:2]
        y_shift = int(height * shift_rate)
        if np.random.rand() < 0.5:
            y_shift = -y_shift
        transformation_matrix = np.float32([[1, 0, 0], [0, 1, y_shift]])
        shifted_image = cv2.warpAffine(image, transformation_matrix, (width, height))
        return shifted_image
