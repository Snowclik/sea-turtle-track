from tools.model_comparator.adapters.base_adapter import ModelAdapter, YoloDetectionAdapter, YoloSegmentationAdapter, UnknownAdapter
from tools.model_comparator.core.report_model import ModelReport

class AdapterFactory:
    @staticmethod
    def get_adapter(report: ModelReport) -> ModelAdapter:
        # Hierarchical discovery
        task = report.task
        
        if task == "detect":
            return YoloDetectionAdapter()
        elif task == "segment":
            return YoloSegmentationAdapter()
            
        # Fallback based on geometry
        if len(report.outputs) == 1:
            if report.outputs[0].shape and report.outputs[0].shape[-1] == 6:
                return YoloDetectionAdapter()
        elif len(report.outputs) == 2:
            return YoloSegmentationAdapter()
            
        return UnknownAdapter()
