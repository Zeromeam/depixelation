from torchvision import transforms
import torch
from PIL import Image
import numpy as np
from typing import Sequence, Union
from torch.utils.data import Dataset
from os import path
from glob import glob

def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    if pil_image.ndim == 2:
        return pil_image.copy()[None]
    if pil_image.ndim != 3:
        raise ValueError("image must have either shape (H, W) or (H, W, 3)")
    if pil_image.shape[2] != 3:
        raise ValueError(f"image has shape (H, W, {pil_image.shape[2]}), but it should have (H, W, 3)")
    
    rgb = pil_image / 255
    rgb_linear = np.where(
        rgb < 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    grayscale_linear = 0.2126 * rgb_linear[..., 0] + 0.7152 * rgb_linear[..., 1] + 0.0722 * rgb_linear[..., 2]
    
    grayscale = np.where(
        grayscale_linear < 0.0031308,
        12.92 * grayscale_linear,
        1.055 * grayscale_linear ** (1 / 2.4) - 0.055
    )
    grayscale = grayscale * 255
    
    if np.issubdtype(pil_image.dtype, np.integer):
        grayscale = np.round(grayscale)
    return grayscale.astype(pil_image.dtype)[None]

def prepare_image(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim < 3 or image.shape[-3] != 1:
        # This is actually more general than the assignment specification
        raise ValueError("image must have shape (..., 1, H, W)")
    if width < 2 or height < 2 or size < 2:
        raise ValueError("width/height/size must be >= 2")
    if x < 0 or (x + width) > image.shape[-1]:
        raise ValueError(f"x={x} and width={width} do not fit into the image width={image.shape[-1]}")
    if y < 0 or (y + height) > image.shape[-2]:
        raise ValueError(f"y={y} and height={height} do not fit into the image height={image.shape[-2]}")
    
    # The (height, width) slices to extract the area that should be pixelated. Since we
    # need this multiple times, specify the slices explicitly instead of using [:] notation
    area = (..., slice(y, y + height), slice(x, x + width))
    
    # This returns already a copy, so we are independent of "image"
    pixelated_image = pixelate(image, x, y, width, height, size)
    
    known_array = np.ones_like(image, dtype=bool)
    known_array[area] = False
    
    # Create a copy to avoid that "target_array" and "image" point to the same array
    target_array = image[area].copy()
    
    return pixelated_image, known_array, target_array


def pixelate(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> np.ndarray:
    # Need a copy since we overwrite data directly
    image = image.copy()
    curr_x = x
    
    while curr_x < x + width:
        curr_y = y
        while curr_y < y + height:
            block = (..., slice(curr_y, min(curr_y + size, y + height)), slice(curr_x, min(curr_x + size, x + width)))
            image[block] = image[block].mean()
            curr_y += size
        curr_x += size
    
    return image

class RandomImagePixelationDataset(Dataset):
    
    def __init__(
            self,
            image: Image,
            width_range: tuple[int, int],
            height_range: tuple[int, int],
            size_range: tuple[int, int],
            dtype = None
    ):
        RandomImagePixelationDataset._check_range(width_range, "width")
        RandomImagePixelationDataset._check_range(height_range, "height")
        RandomImagePixelationDataset._check_range(size_range, "size")
        self.image = image
        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range
        self.dtype = dtype
    
    @staticmethod
    def _check_range(r: tuple[int, int], name: str):
        if r[0] < 2:
            raise ValueError(f"minimum {name} must be >= 2")
        if r[0] > r[1]:
            raise ValueError(f"minimum {name} must be <= maximum {name}")
    
    def __getitem__(self, index):
        sample = self.image
        im = np.array(sample, dtype=self.dtype)
        # im = to_grayscale(np_im)  # Image shape is now (1, H, W)
        image_width = im.shape[-1]
        image_height = im.shape[-2]
        
        # Create RNG in each __getitem__ call to ensure reproducibility even in
        # environments with multiple threads and/or processes
        rng = np.random.default_rng(seed=index)
        
        # Both width and height can be arbitrary, but they must not exceed the
        # actual image width and height
        width = min(rng.integers(low=self.width_range[0], high=self.width_range[1], endpoint=True), image_width)
        height = min(rng.integers(low=self.height_range[0], high=self.height_range[1], endpoint=True), image_height)
        
        # Ensure that x and y always fit with the randomly chosen width and
        # height (and not throw an error in "prepare_image")
        x = rng.integers(image_width - width, endpoint=True)
        y = rng.integers(image_height - height, endpoint=True)
        
        # Block size can be arbitrary again
        size = rng.integers(low=self.size_range[0], high=self.size_range[1], endpoint=True)
        pixelated_image, known_array, target_array = prepare_image(im, x, y, width, height, size)
        return pixelated_image, known_array, target_array, im
    
    # def __len__(self):
    #     return len(self.dataset)
    


def random_augmented_image(image: Image,image_size: Union[int, Sequence[int]],augment: bool,seed: int) -> torch.Tensor:
    torch.manual_seed(seed)

    im_shape = 64
    #print(augment)
    if augment :
        transform_list = [
        transforms.RandomAutocontrast(),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.ColorJitter(brightness=.3, hue=.2)
        ]
    
        indices = torch.randperm(len(transform_list))[:2]
        choosen_list = []
        for i in indices :
            choosen_list.append(transform_list[i])
        transform_chain = transforms.Compose([
                transforms.RandomEqualize(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=.3, hue=.2),
                transforms.RandomAutocontrast(),    
                transforms.RandomAdjustSharpness(sharpness_factor=2),   
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.RandomResizedCrop(size=im_shape,scale=(.5,1)),
                # transforms.Resize(size=im_shape),
                # transforms.CenterCrop(size=(im_shape,im_shape)),
                
                #*choosen_list,
                transforms.Grayscale(),
                #torch.nn.Dropout()
                transforms.ToTensor()
                ])
    else:
                transform_chain = transforms.Compose([
                transforms.Resize(size=im_shape),
                transforms.CenterCrop(size=(im_shape,im_shape)),
                transforms.Grayscale(),
                transforms.ToTensor()
                ])
    im = transform_chain(image)*255
    #trans = transforms.ToTensor()
    return im


class TransformedImageDataset(Dataset):
    
    def __init__(self, dataset,seed,augment:bool = True):
        """
        Apply random transform chains to a dataset.
        """
        self.dataset = dataset
        self.seed = seed
        self.augment = augment 
    
    def __getitem__(self, index: int):
        """
        Return a transformed image from the original dataset.
        """
        sample = self.dataset[index]

        pixlated_im = random_augmented_image(sample  ,sample.size,self.augment, seed= self.seed)
        pixlated_im_fun = RandomImagePixelationDataset(pixlated_im,(4,32), (4,32),(4,16))
        pixelated_image, known_array, target_array, original_im = pixlated_im_fun[index]
        #print(pixelated_image.shape, known_array.shape, target_array.shape, original_im)
        return pixelated_image, known_array, original_im

    def __len__(self):
        return len(self.dataset)