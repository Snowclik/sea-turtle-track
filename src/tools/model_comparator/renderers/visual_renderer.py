import cv2
import numpy as np

class BaseRenderer:
    @staticmethod
    def render(image: np.ndarray, post_data: dict, classes: dict) -> np.ndarray:
        return image.copy()

class BBoxRenderer(BaseRenderer):
    @staticmethod
    def render(image: np.ndarray, post_data: dict, classes: dict) -> np.ndarray:
        img = image.copy()
        bboxes = post_data.get("bboxes", [])
        scores = post_data.get("scores", [])
        
        # Escalar
        h, w = img.shape[:2]
        
        for box, score in zip(bboxes, scores):
            if score > 0.1:
                # YOLOv8 ONNX end2end exports normalized coords usually? Wait, if they are not normalized, we don't scale.
                # Actually, output is usually pixel coords relative to input_shape.
                x1, y1, x2, y2 = map(int, box[:4])
                # We assume input is 640x640, we must scale to original image. For simplicity, assume image is resized to 640x640 before rendering.
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return img

class PolygonRenderer(BaseRenderer):
    @staticmethod
    def render(image: np.ndarray, post_data: dict, classes: dict) -> np.ndarray:
        img = image.copy()
        masks = post_data.get("masks", None)
        scores = post_data.get("scores", [])
        
        if masks is None:
            return BBoxRenderer.render(image, post_data, classes)
            
        h, w = img.shape[:2]
        
        for i, (mask, score) in enumerate(zip(masks, scores)):
            if score > 0.1:
                mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        return img

class MosaicRenderer:
    @staticmethod
    def side_by_side(img_a: np.ndarray, img_b: np.ndarray, title_a="Model A", title_b="Model B") -> np.ndarray:
        hA, wA = img_a.shape[:2]
        hB, wB = img_b.shape[:2]
        
        target_h = max(hA, hB)
        
        a_resized = cv2.resize(img_a, (int(wA * target_h / hA), target_h))
        b_resized = cv2.resize(img_b, (int(wB * target_h / hB), target_h))
        
        # Add labels
        cv2.putText(a_resized, title_a, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(b_resized, title_b, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        return np.hstack((a_resized, b_resized))
