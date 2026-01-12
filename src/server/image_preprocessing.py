"""
Image Preprocessing Module for Plant Disease Detection Pipeline.

This module handles image segmentation using OpenCV techniques including:
- GrabCut algorithm for foreground extraction
- Contour detection for leaf boundary extraction
- Mask generation for background removal

The preprocessing is designed to isolate the leaf from the background,
improving classification accuracy by removing noise and irrelevant elements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import io

import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class PreprocessingResult:
    """Result of image preprocessing."""
    processed_image: Image.Image
    mask: Optional[np.ndarray] = None
    contour_points: Optional[np.ndarray] = None
    preprocessing_applied: str = "none"


class ImagePreprocessor:
    """
    Image preprocessor using OpenCV for leaf segmentation.
    
    Pipeline:
    1. GrabCut for foreground/background separation
    2. Mask refinement using morphological operations
    3. Contour extraction for leaf boundary
    4. Apply mask to original image
    """

    DEFAULT_GRABCUT_ITERATIONS = 5
    DEFAULT_MORPH_KERNEL_SIZE = 5

    def __init__(
        self,
        grabcut_iterations: int = DEFAULT_GRABCUT_ITERATIONS,
        morph_kernel_size: int = DEFAULT_MORPH_KERNEL_SIZE,
        use_green_mask: bool = True,
    ):
        """
        Initialize the preprocessor.
        
        Args:
            grabcut_iterations: Number of iterations for GrabCut algorithm
            morph_kernel_size: Kernel size for morphological operations
            use_green_mask: Whether to use green color masking as additional hint
        """
        if not CV2_AVAILABLE:
            raise ImportError(
                "OpenCV (cv2) is required for image preprocessing. "
                "Install it with: pip install opencv-python"
            )
        
        self.grabcut_iterations = grabcut_iterations
        self.morph_kernel_size = morph_kernel_size
        self.use_green_mask = use_green_mask

    def _pil_to_cv2(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV BGR format."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        rgb_array = np.array(image)
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    def _cv2_to_pil(self, image: np.ndarray) -> Image.Image:
        """Convert OpenCV BGR image to PIL Image."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(rgb, mode="RGBA")
        else:
            rgb = image
        return Image.fromarray(rgb)

    def _create_initial_mask_from_green(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Create an initial mask based on green color detection.
        Plants/leaves are typically green, so this helps GrabCut.
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        
        # Green color range in HSV
        # Hue: 35-85 (green range), Saturation: 40-255, Value: 40-255
        lower_green = np.array([25, 30, 30])
        upper_green = np.array([95, 255, 255])
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Also include some brown/yellow for diseased leaves
        lower_yellow = np.array([15, 30, 30])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(green_mask, yellow_mask)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask

    def _apply_grabcut(
        self,
        image_bgr: np.ndarray,
        initial_mask: Optional[np.ndarray] = None,
        rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        Apply GrabCut algorithm for foreground extraction.
        
        Args:
            image_bgr: Input image in BGR format
            initial_mask: Optional initial mask (from green color detection)
            rect: Optional bounding rectangle (x, y, w, h)
            
        Returns:
            Binary mask where foreground is 255, background is 0
        """
        h, w = image_bgr.shape[:2]
        
        # Initialize GrabCut mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Background and foreground models (required by GrabCut)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        if initial_mask is not None and self.use_green_mask:
            # Use color-based mask as initialization
            # GrabCut mask values:
            # 0: sure background, 1: sure foreground
            # 2: probable background, 3: probable foreground
            mask[initial_mask == 0] = cv2.GC_PR_BGD  # probable background
            mask[initial_mask > 0] = cv2.GC_PR_FGD   # probable foreground
            
            # Set margins as definite background
            margin = 10
            mask[:margin, :] = cv2.GC_BGD
            mask[-margin:, :] = cv2.GC_BGD
            mask[:, :margin] = cv2.GC_BGD
            mask[:, -margin:] = cv2.GC_BGD
            
            mode = cv2.GC_INIT_WITH_MASK
            rect = None
        else:
            # Use rectangle mode if no initial mask
            if rect is None:
                # Use center rectangle covering 80% of image
                margin_x = int(w * 0.1)
                margin_y = int(h * 0.1)
                rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
            mode = cv2.GC_INIT_WITH_RECT
        
        try:
            cv2.grabCut(
                image_bgr,
                mask,
                rect,
                bgd_model,
                fgd_model,
                self.grabcut_iterations,
                mode
            )
        except cv2.error:
            # GrabCut failed, return full image mask
            return np.ones((h, w), dtype=np.uint8) * 255
        
        # Create binary mask from GrabCut result
        # Foreground if definitely foreground (1) or probably foreground (3)
        binary_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype(np.uint8)
        
        return binary_mask

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """Refine mask using morphological operations."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        
        # Close small holes
        refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Remove small noise
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return refined

    def _find_largest_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Find the largest contour in the mask."""
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Find largest contour by area
        largest = max(contours, key=cv2.contourArea)
        return largest

    def _apply_mask(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        fill_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """Apply mask to image, filling background with specified color."""
        # Create output image with fill color
        result = np.full_like(image_bgr, fill_color)
        
        # Copy foreground from original image
        result[mask > 0] = image_bgr[mask > 0]
        
        return result

    def preprocess(
        self,
        image: Image.Image,
        apply_mask: bool = True,
        fill_background: Tuple[int, int, int] = (255, 255, 255),
    ) -> PreprocessingResult:
        """
        Apply full preprocessing pipeline to segment the leaf.
        
        Pipeline:
        1. Convert to OpenCV format
        2. Create initial mask from green color detection
        3. Apply GrabCut for refined segmentation
        4. Refine mask with morphological operations
        5. Find largest contour (the leaf)
        6. Apply mask to original image
        
        Args:
            image: Input PIL Image
            apply_mask: Whether to apply the mask to the image
            fill_background: Background color (BGR) when mask is applied
            
        Returns:
            PreprocessingResult with processed image and metadata
        """
        image_bgr = self._pil_to_cv2(image)
        
        # Step 1: Create initial mask from color
        initial_mask = self._create_initial_mask_from_green(image_bgr)
        
        # Step 2: Apply GrabCut
        grabcut_mask = self._apply_grabcut(image_bgr, initial_mask)
        
        # Step 3: Refine mask
        refined_mask = self._refine_mask(grabcut_mask)
        
        # Step 4: Find largest contour
        contour = self._find_largest_contour(refined_mask)
        
        # If we have a contour, create a clean mask from it
        if contour is not None and cv2.contourArea(contour) > 100:
            h, w = image_bgr.shape[:2]
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            
            # Use contour mask if it's valid
            final_mask = contour_mask
            contour_points = contour
        else:
            final_mask = refined_mask
            contour_points = None
        
        # Step 5: Apply mask to image if requested
        if apply_mask:
            # Convert fill color from RGB to BGR
            fill_bgr = (fill_background[2], fill_background[1], fill_background[0])
            result_bgr = self._apply_mask(image_bgr, final_mask, fill_bgr)
            processed_image = self._cv2_to_pil(result_bgr)
            preprocessing_type = "grabcut_masked"
        else:
            processed_image = image
            preprocessing_type = "grabcut_mask_only"
        
        return PreprocessingResult(
            processed_image=processed_image,
            mask=final_mask,
            contour_points=contour_points,
            preprocessing_applied=preprocessing_type
        )

    def segment_leaf(
        self,
        image: Image.Image,
        return_with_transparent_bg: bool = False
    ) -> Image.Image:
        """
        Simplified method to segment leaf and return the result.
        
        Args:
            image: Input PIL Image
            return_with_transparent_bg: If True, return RGBA with transparent background
            
        Returns:
            Processed PIL Image
        """
        result = self.preprocess(image, apply_mask=True)
        
        if return_with_transparent_bg and result.mask is not None:
            # Convert to RGBA with transparent background
            image_bgr = self._pil_to_cv2(image)
            image_rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
            image_rgba[:, :, 3] = result.mask  # Set alpha channel
            return self._cv2_to_pil(image_rgba)
        
        return result.processed_image


def create_preprocessor(
    grabcut_iterations: int = 5,
    morph_kernel_size: int = 5,
    use_green_mask: bool = True,
) -> ImagePreprocessor:
    """Factory function to create an ImagePreprocessor."""
    return ImagePreprocessor(
        grabcut_iterations=grabcut_iterations,
        morph_kernel_size=morph_kernel_size,
        use_green_mask=use_green_mask,
    )
