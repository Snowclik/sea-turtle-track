from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class PerformanceMetrics:
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    fps: float = 0.0
    ram_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

@dataclass
class OutputMetrics:
    num_detections: int = 0
    avg_confidence: float = 0.0
    num_polygons: int = 0
    mask_coverage_percent: float = 0.0

@dataclass
class TensorInfo:
    name: str
    shape: List[int]
    dtype: str

@dataclass
class ModelReport:
    # Identity
    model_name: str = "Unknown"
    file_path: str = ""
    size_mb: float = 0.0
    sha256: str = ""
    
    # Architecture & Metadata
    framework: str = "Unknown"
    framework_version: str = "Unknown"
    export_version: str = "Unknown"
    architecture: str = "Unknown"
    task: str = "Unknown"
    onnx_opset: int = 0
    num_nodes: int = 0
    num_parameters: Optional[int] = None
    classes: Dict[int, str] = field(default_factory=dict)
    
    # Tensors
    inputs: List[TensorInfo] = field(default_factory=list)
    outputs: List[TensorInfo] = field(default_factory=list)
    
    # Execution
    execution_provider: str = ""
    
    # Telemetry
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    results: OutputMetrics = field(default_factory=OutputMetrics)
