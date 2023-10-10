# PyTorch imports:
import torch
import torchvision.transforms as T


class Sharpen():
    def __init__(self, sharpness_factor):
        self.sharpness_factor = sharpness_factor

    def __call__(self, image : torch.tensor) -> torch.tensor:
        transformed_image = T.functional.adjust_sharpness(image, self.sharpness_factor)

        return transformed_image

class Contrast():
    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, image : torch.tensor) -> torch.tensor:
        transformed_image = T.functional.adjust_contrast(image, self.contrast_factor)

        return transformed_image
