# module tiền xử lí ảnh cho pipeline nhận diện bệnh cây
# module này xử lí tách nền/lá bằng các kĩ thuật opencv như:
# - grabcut để tách foreground
# - tìm contour để lấy biên lá
# - tạo mask để xoá nền
# mục tiêu là cô lập vùng lá khỏi nền để giảm nhiễu
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
    # kết quả tiền xử lí ảnh
    processed_image: Image.Image
    mask: Optional[np.ndarray] = None
    contour_points: Optional[np.ndarray] = None
    preprocessing_applied: str = "none"


class ImagePreprocessor:
    # tiền xử lí ảnh bằng opencv để tách vùng lá
    # pipeline:
    # 1) grabcut để tách foreground/background
    # 2) refine mask bằng phép biến đổi hình thái
    # 3) lấy contour biên lá
    # 4) áp mask lên ảnh gốc

    DEFAULT_GRABCUT_ITERATIONS = 5
    DEFAULT_MORPH_KERNEL_SIZE = 5

    def __init__(
        self,
        grabcut_iterations: int = DEFAULT_GRABCUT_ITERATIONS,
        morph_kernel_size: int = DEFAULT_MORPH_KERNEL_SIZE,
        use_green_mask: bool = True,
    ):
        # khởi tạo preprocessor
        # grabcut_iterations: số vòng lặp cho grabcut
        # morph_kernel_size: kích thước kernel cho phép hình thái
        # use_green_mask: có dùng mask màu xanh làm gợi ý bổ sung hay không
        if not CV2_AVAILABLE:
            raise ImportError(
                "OpenCV (cv2) is required for image preprocessing. "
                "Install it with: pip install opencv-python"
            )
        
        self.grabcut_iterations = grabcut_iterations
        self.morph_kernel_size = morph_kernel_size
        self.use_green_mask = use_green_mask

    def _pil_to_cv2(self, image: Image.Image) -> np.ndarray:
        # đổi ảnh PIL sang mảng opencv (BGR)
        if image.mode != "RGB":
            image = image.convert("RGB")
        rgb_array = np.array(image)
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    def _cv2_to_pil(self, image: np.ndarray) -> Image.Image:
        # đổi mảng opencv (BGR/BGRA) về ảnh PIL
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(rgb, mode="RGBA")
        else:
            rgb = image
        return Image.fromarray(rgb)

    def _create_initial_mask_from_green(self, image_bgr: np.ndarray) -> np.ndarray:
        # tạo mask khởi tạo dựa trên màu xanh
        # lá cây thường xanh (và có thể vàng/nâu khi bệnh), giúp grabcut hội tụ tốt hơn
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
        # chạy grabcut để tách foreground
        # trả về binary mask: foreground=255, background=0
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
        # refine mask bằng phép biến đổi hình thái
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
        # tìm contour lớn nhất trong mask
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
        # áp mask lên ảnh, vùng nền sẽ được fill bằng màu chỉ định
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
        # chạy full pipeline để tách vùng lá
        # các bước:
        # 1) chuyển sang opencv
        # 2) tạo mask khởi tạo theo màu
        # 3) chạy grabcut
        # 4) refine mask
        # 5) tìm contour lớn nhất (giả định là lá)
        # 6) áp mask lên ảnh gốc
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
        # hàm gọn để tách lá và trả về ảnh kết quả
        # return_with_transparent_bg=True sẽ trả RGBA với nền trong suốt
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
    # factory để tạo ImagePreprocessor
    return ImagePreprocessor(
        grabcut_iterations=grabcut_iterations,
        morph_kernel_size=morph_kernel_size,
        use_green_mask=use_green_mask,
    )
