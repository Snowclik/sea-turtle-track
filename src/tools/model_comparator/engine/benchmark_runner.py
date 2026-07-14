import onnxruntime as ort
import numpy as np
import time
import psutil
from tools.model_comparator.core.report_model import ModelReport
from tools.model_comparator.adapters.base_adapter import ModelAdapter

class BenchmarkRunner:
    @staticmethod
    def run(report: ModelReport, adapter: ModelAdapter, image: np.ndarray, num_runs: int = 5):
        session = ort.InferenceSession(report.file_path, providers=['CPUExecutionProvider'])
        report.execution_provider = session.get_providers()[0]
        
        # Prepare input
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        # Handle dynamic batch size if present
        if isinstance(input_shape[0], str) or input_shape[0] < 1:
            input_shape[0] = 1
        
        tensor = adapter.preprocess(image, input_shape)
        
        # Warmup
        for _ in range(3):
            session.run(None, {input_name: tensor})
            
        latencies = []
        process = psutil.Process()
        
        mem_before = process.memory_info().rss
        cpu_percentages = []
        
        # We need the output tensors for postprocessing later
        final_outputs = None
        
        for _ in range(num_runs):
            t0 = time.time()
            cpu_before = process.cpu_percent(interval=None)
            outputs = session.run(None, {input_name: tensor})
            cpu_percentages.append(process.cpu_percent(interval=None))
            latencies.append((time.time() - t0) * 1000)
            final_outputs = outputs
            
        mem_after = process.memory_info().rss
        
        report.performance.avg_latency_ms = float(np.mean(latencies))
        report.performance.min_latency_ms = float(np.min(latencies))
        report.performance.max_latency_ms = float(np.max(latencies))
        report.performance.std_latency_ms = float(np.std(latencies))
        report.performance.fps = 1000.0 / report.performance.avg_latency_ms if report.performance.avg_latency_ms > 0 else 0.0
        
        report.performance.ram_usage_mb = (mem_after - mem_before) / (1024 * 1024)
        report.performance.cpu_usage_percent = float(np.mean(cpu_percentages))
        
        # Run postprocess to get metrics
        post_data = adapter.postprocess(final_outputs, list(image.shape))
        
        # Output Metrics
        scores = post_data.get("scores", np.array([]))
        masks = post_data.get("masks", None)
        valid_dets = scores[scores > 0.1]
        
        report.results.num_detections = len(valid_dets)
        report.results.avg_confidence = float(np.mean(valid_dets)) if len(valid_dets) > 0 else 0.0
        
        if masks is not None and len(valid_dets) > 0:
            import cv2
            polys = 0
            # Just approximation for metric
            for i in range(len(valid_dets)):
                c, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polys += len(c)
            report.results.num_polygons = polys
            
        return post_data
