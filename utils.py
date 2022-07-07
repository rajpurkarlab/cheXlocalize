import numpy as np
from pycocotools import mask


def encode_segmentation(segmentation_arr):
    """
    Encode a binary segmentation (np.array) to RLE format using the pycocotools Mask API.
    Args:
        segmentation_arr (np.array): [h x w] binary segmentation
    Returns:
		Rs (dict): the encoded mask in RLE format
    """
    segmentation = np.asfortranarray(segmentation_arr.astype('uint8'))
    Rs = mask.encode(segmentation)
    Rs['counts'] = Rs['counts'].decode()
    return Rs
