"""Image quality and drift feature extraction."""

from pathlib import Path
from typing import Dict, Union
import numpy as np
from PIL import Image
from scipy.signal import convolve2d


class DriftFeatureExtractor:
    """Extracts low-level image features for drift and shift detection."""

    def __init__(self):
        # 3x3 Laplacian high-pass filter
        self.laplacian_kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)

        # Sobel gradient filters
        self.sobel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32) / 4.0

        self.sobel_y = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=np.float32) / 4.0

    def extract_from_array(self, img_array: np.ndarray) -> Dict[str, float]:
        """Compute 5 low-level image drift features from a grayscale or RGB numpy array."""
        # Convert to 2D grayscale if RGB
        if img_array.ndim == 3:
            # Luminance weighted conversion: 0.2989 R + 0.5870 G + 0.1140 B
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array.astype(np.float32)

        # If array is in [0, 255], normalize to [0, 1]
        if gray.max() > 1.0:
            norm_gray = gray / 255.0
            unnorm_gray = gray
        else:
            norm_gray = gray
            unnorm_gray = gray * 255.0

        brightness = float(np.mean(norm_gray))
        contrast = float(np.std(norm_gray))

        # Sobel edge gradient magnitude
        gx = convolve2d(norm_gray, self.sobel_x, mode="valid")
        gy = convolve2d(norm_gray, self.sobel_y, mode="valid")
        edge_strength = float(np.mean(np.sqrt(gx**2 + gy**2)))

        # High-frequency Laplacian features on 0-255 scale
        lap = convolve2d(unnorm_gray, self.laplacian_kernel, mode="valid")
        high_freq_std = float(np.std(lap))
        high_freq_mean_abs = float(np.mean(np.abs(lap)))

        return {
            "brightness": brightness,
            "contrast": contrast,
            "edge_strength": edge_strength,
            "high_frequency_std": high_freq_std,
            "high_frequency_mean_abs": high_freq_mean_abs
        }

    def extract_from_image(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, float]:
        """Load image and extract drift features."""
        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as img:
                arr = np.array(img.convert("RGB"), dtype=np.float32)
        elif isinstance(image_input, Image.Image):
            arr = np.array(image_input.convert("RGB"), dtype=np.float32)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        return self.extract_from_array(arr)
