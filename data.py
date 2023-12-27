import os
import cv2
import h5py
import torch
import numpy as np
import albumentations as A

from dataclasses import dataclass
from torchvision.transforms import Resize
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, TypeVar, List, Callable, Iterable


T = TypeVar("T")


DEFAULT_TRANSFORM = A.Compose([
    A.Resize(256,256, interpolation=cv2.INTER_NEAREST),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
    A.RandomCrop(height=256, width=256, always_apply=True),
    A.RandomBrightness(p=1),
    A.OneOf(
        [
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ],
        p=0.9,
    ),

])

VAL_TRANSFORM = A.Compose([
    A.Resize(256,256, interpolation=cv2.INTER_NEAREST),
])


def get_image_and_mask(list_file: str, prefix:str=".") -> Iterable[Tuple[str, str]]:
    with open(list_file, "r") as f:
        # skip first 6 row
        for _ in range(6):
            next(f)
        for line in f:
            image_name = line.split(" ")[0]
            image_path = os.path.join(prefix, "images", "images", f"{image_name}.jpg")
            image_mask = os.path.join(prefix, "annotations", "annotations", "trimaps", f"{image_name}.png")
            yield image_path, image_mask


@dataclass
class Augmenter:
    transform: A.Compose

    def augment_image(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        image_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.numpy()

        augmented = self.transform(image=image_np, mask=mask_np)
        augmented_image, augmented_mask = augmented['image'], augmented['mask']

        augmented_image = torch.tensor(augmented_image, dtype=torch.float32).permute(2, 0, 1)
        augmented_mask = torch.tensor(augmented_mask, dtype=torch.float32)

        return augmented_image, augmented_mask


# Define type aliases for preprocess functions
PreprocessFunction = Callable[[str], torch.Tensor]


def default_preprocess_image(path: str) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype('float32') 
    mx = np.max(img)
    if mx:
        img/=mx 
    img = np.transpose(img, (2, 0, 1))
    img_ten = torch.tensor(img)
    return img_ten
    

def default_preprocess_mask(path: str) -> torch.Tensor:
    msk = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    msk[msk==2] = 0
    msk[msk > 0] = 255
    msk = msk.astype('float32')
    msk /= 255.0
    msk_ten = torch.tensor(msk)
    return msk_ten


class SegmentationDataset(Dataset):
    def __init__(self, 
        images_file: List[str], 
        masks_file: List[str], 
        preprocess_image_fn: PreprocessFunction = default_preprocess_image,
        preprocess_mask_fn: PreprocessFunction = default_preprocess_mask,
        augmentation_transforms: Callable[[T], T] = Augmenter(DEFAULT_TRANSFORM).augment_image
    ):
        self.images_file = images_file
        self.masks_file = masks_file
        self.augmentation_transforms = augmentation_transforms
        self.preprocess_image_fn = preprocess_image_fn
        self.preprocess_mask_fn = preprocess_mask_fn

    def __len__(self):
        return len(self.images_file)

    def __getitem__(self, idx):
       
        image_file = self.images_file[idx]
        mask_file = self.masks_file[idx]

        image = self.preprocess_image_fn(image_file)
        mask = self.preprocess_mask_fn(mask_file)

        if self.augmentation_transforms:
            image, mask = self.augmentation_transforms(image, mask)

        return image, mask
    
    def get_examples(self, start=0, end=10):
        image_files = self.images_file[start:end]
        mask_files = self.masks_file[start:end]

        resize_fn =  Resize((256,256), interpolation=cv2.INTER_NEAREST)

        images = []
        masks = []
        for image_file, mask_file in zip(image_files, mask_files):
            image = resize_fn(self.preprocess_image_fn(image_file))
            mask = resize_fn(torch.unsqueeze(self.preprocess_mask_fn(mask_file), 0))
            images.append(image)
            masks.append(mask)
        return torch.stack(images, dim=0), torch.concat(masks, dim=0)


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def subset_preprocess_mask(lre, H=1303, W=912):
    mask = rle_decode(lre, (H, W))
    mask_tensor = torch.Tensor(mask)
    resized_mask_tensor = Resize((H//2, W//2))(torch.unsqueeze(mask_tensor, 0)).squeeze()
    return resized_mask_tensor


class H5ImageProcess:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
    

    def _transform(self, image_arr):
        image_arr = image_arr.astype(np.float32)
        image_arr = np.tile(image_arr[...,None],[1, 1, 3]) 
        mx = np.max(image_arr)
        if mx:
            image_arr/=mx 
        img = np.transpose(image_arr, (2, 0, 1))
        img_ten = torch.tensor(img)
        return img_ten
    
    def preprocess_image_train(self, idx):
        with h5py.File(self.dataset_path, "r") as f:
            arr = f["kidney_1_dense/arr"][idx]
            min_value = f["kidney_1_dense/arr"][idx]
        arr = arr + min_value
        return self._transform(arr)
    
    def preprocess_image_val(self, idx):
        with h5py.File(self.dataset_path, "r") as f:
            arr = f["kidney_3_dense/arr"][idx]
            min_value = f["kidney_3_dense/arr"][idx]
        arr = arr + min_value
        return self._transform(arr)

def main():
    image_files, label_files = tuple(zip(*list(get_image_and_mask("data/annotations/annotations/list.txt", prefix="./data"))))
    train_image_files, val_image_files, train_mask_files, val_mask_files = train_test_split(
        image_files, label_files, test_size=0.2, random_state=42)
    augmenter = Augmenter(DEFAULT_TRANSFORM)

    train_dataset = SegmentationDataset(train_image_files, train_mask_files, augmentation_transforms=augmenter.augment_image)
    val_dataset = SegmentationDataset(val_image_files, val_mask_files, augmentation_transforms=augmenter.augment_image)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    for batch_idx, (images, masks) in enumerate(train_dataloader):
        print(images.shape)
        print(masks.shape)
        break

if __name__ == "__main__":
    main()

