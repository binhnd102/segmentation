import numpy as np
import torch

from PIL import Image
from typing import Protocol, List
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage, ToTensor
from lightning.pytorch.callbacks import Callback


def mask_overlay(
        rgb_image: Image, 
        mask_image: Image
        ) -> Image:
    mask_image = mask_image.convert("RGB")

    rgb_array = np.array(rgb_image)
    mask_array = np.array(mask_image)

    opacity = 0.7  # Adjust opacity value as desired (0.0 to 1.0)
    mask_array = mask_array / 255.0  # Normalize mask values to 0-1 range
    overlay_array = rgb_array * (1 - opacity) + mask_array * opacity * 255  # Blend with opacity

    overlay_image = Image.fromarray(overlay_array.astype(np.uint8))
    return overlay_image


def visualize_segmentation(
        images_batch: torch.Tensor, 
        outputs_batch: torch.Tensor
        ) -> Image:
    image = ToPILImage()(make_grid(images_batch, nrow=5))
    mask = ToPILImage()(make_grid(torch.sigmoid(outputs_batch), nrow=5))
    overlay = mask_overlay(image, mask)
    return overlay


class VisualizeValDataCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch = 0


    def on_validation_epoch_end(self, trainer, pl_module):
        device = trainer.model.device
        val_dataset = trainer.val_dataloaders.dataset

        # Get a random batch of validation data
        images, masks = val_dataset.get_examples()

        # Run the model on the batch
        with torch.no_grad():
            model_output = pl_module.model(images.to(device))  # Assuming input is at index 0

        visualization = visualize_segmentation(images.data.cpu(), model_output.data.cpu())

        logger = trainer.logger
        logger.log_image(key="val_image", images=[ToTensor()(visualization)])
        
        self.epoch += 1