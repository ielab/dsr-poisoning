from typing import Union, Optional
from transformers.image_utils import (
    ChannelDimension,
)
import torch
import torch.nn.functional as F
import os, json


def store_results(results, path, dset_name=""):
    # Store the results in a json file in a dictionary format
    # Append if the file already exists; create a new file if it doesn't
    path_to_file = os.path.join(path, f'eval_results_{dset_name}.json')
    if os.path.exists(path_to_file):
        with open(path_to_file, 'r') as f:
            data = json.load(f)
        data.update(results)
    else:
        data = results
    with open(path_to_file, 'w') as f:
        json.dump(data, f)
    return


def add_margin(image, margin_ratio, init_black=True):
    """
    Resizes the original RGB image to fit within a margin of fixed ratio,
    ensuring the overall output dimensions match the original image's dimensions.

    Args:
        image (torch.Tensor): Original RGB image (H, W, C).
        margin_ratio (float): Ratio of the margin to the image dimensions (0 <= margin_ratio <= 1).

    Returns:
        torch.Tensor: Image with black margin, same size as original (H, W, C).
        torch.Tensor: Mask where 1 indicates margin pixels (H, W, 1).
    """
    if not (0 <= margin_ratio <= 1):
        raise ValueError("Margin ratio must be between 0 and 1 inclusive.")

    H, W, C = image.shape
    margin_height = int(margin_ratio/2 * H)
    margin_width = int(margin_ratio/2 * W)

    # Handle the case where the margin covers the entire image
    if margin_ratio == 1.0:
        if init_black:
            image_with_margin = torch.zeros((H, W, C), dtype=image.dtype, device=image.device)
        else:
            image_with_margin = torch.ones((H, W, C), dtype=image.dtype, device=image.device) * 255
        mask = torch.ones((H, W, C), dtype=torch.bool, device=image.device)
        return image_with_margin, mask

    inner_H, inner_W = H - 2 * margin_height, W - 2 * margin_width

    if inner_H <= 0 or inner_W <= 0:
        raise ValueError("Margin ratio is too large, resulting in an unusable resized image.")

    # Resize the image to fit inside the margin
    resized_image = F.interpolate(
        image.permute(2, 0, 1).unsqueeze(0),  # Change to (N, C, H, W) for resizing
        size=(inner_H, inner_W),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).permute(1, 2, 0)  # Back to (H, W, C)

    # Create the output tensor with black margin
    if init_black:
        image_with_margin = torch.zeros((H, W, C), dtype=image.dtype, device=image.device)
    else:
        image_with_margin = torch.ones((H, W, C), dtype=image.dtype, device=image.device) * 255

    # Place the resized image in the center
    image_with_margin[margin_height:-margin_height, margin_width:-margin_width, :] = resized_image

    # Create a mask for the margin area
    mask = torch.ones((H, W, 1), dtype=torch.bool, device=image.device)
    mask[margin_height:-margin_height, margin_width:-margin_width, :] = False
    mask = mask.expand_as(image_with_margin)
    return image_with_margin, mask


def to_channel_dimension_format(
    image: torch.Tensor,
    channel_dim: Union[ChannelDimension, str],
    input_channel_dim: Optional[Union[ChannelDimension, str]] = None,
) -> torch.Tensor:
    """
    Converts `image` to the channel dimension format specified by `channel_dim`.

    Args:
        image (`torch.Tensor`):
            The image to have its channel dimension set.
        channel_dim (`ChannelDimension`):
            The channel dimension format to use.
        input_channel_dim (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.

    Returns:
        `torch.Tensor`: The image with the channel dimension set to `channel_dim`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input image must be of type np.ndarray, got {type(image)}")

    if input_channel_dim is None:
        raise ValueError("Input channel dimension format must be provided.")

    target_channel_dim = ChannelDimension(channel_dim)
    if input_channel_dim == target_channel_dim:
        return image

    if target_channel_dim == ChannelDimension.FIRST:
        image = image.permute(2, 0, 1)
    elif target_channel_dim == ChannelDimension.LAST:
        image = image.permute(1, 2, 0)
    else:
        raise ValueError("Unsupported channel dimension format: {}".format(channel_dim))

    return image