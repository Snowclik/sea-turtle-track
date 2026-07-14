from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List

class ModelAdapter(ABC):
    @abstractmethod
    def preprocess(self, image: np.ndarray, input_shape: List[int]) -> np.ndarray:
        pass

    @abstractmethod
    def postprocess(self, outputs: List[np.ndarray], img_shape: List[int]) -> Dict[str, Any]:
        pass

class YoloDetectionAdapter(ModelAdapter):
    def preprocess(self, image: np.ndarray, input_shape: List[int]) -> np.ndarray:
        import cv2
        h, w = input_shape[2:]
        img = cv2.resize(image, (w, h))
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def postprocess(self, outputs: List[np.ndarray], img_shape: List[int]) -> Dict[str, Any]:
        # outputs[0] is typically [1, 300, 6] for end2end detect
        out = outputs[0]
        if len(out.shape) == 3:
            out = out[0]
        return {"bboxes": out[:, :4], "scores": out[:, 4], "classes": out[:, 5]}

class YoloSegmentationAdapter(YoloDetectionAdapter):
    def postprocess(self, outputs: List[np.ndarray], img_shape: List[int]) -> Dict[str, Any]:
        # outputs[0] is [1, 300, 38], outputs[1] is [1, 32, 160, 160]
        out = outputs[0]
        if len(out.shape) == 3:
            out = out[0]
        bboxes = out[:, :4]
        scores = out[:, 4]
        classes = out[:, 5]
        
        masks = None
        if len(outputs) > 1:
            coeffs = out[:, 6:]
            protos = outputs[1][0]
            # Simple MatMul for mask generation
            # coeffs: [N, 32], protos: [32, 160*160]
            c, mh, mw = protos.shape
            mask_flat = coeffs @ protos.reshape(c, -1)
            masks_raw = mask_flat.reshape(-1, mh, mw)
            # Sigmoid
            masks_raw = 1 / (1 + np.exp(-masks_raw))
            masks = masks_raw > 0.5
            
        return {"bboxes": bboxes, "scores": scores, "classes": classes, "masks": masks}

class UnknownAdapter(YoloDetectionAdapter):
    pass
