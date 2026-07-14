import onnx
import hashlib
import os
import json
from tools.model_comparator.core.report_model import ModelReport, TensorInfo

class StructuralInspector:
    """Inspecciona el archivo ONNX y extrae su metadata."""
    
    @staticmethod
    def inspect(filepath: str) -> ModelReport:
        report = ModelReport(file_path=filepath)
        report.model_name = os.path.basename(filepath)
        report.size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        # Calculate SHA256
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        report.sha256 = sha256_hash.hexdigest()
        
        try:
            model = onnx.load(filepath)
            report.onnx_opset = model.opset_import[0].version if len(model.opset_import) > 0 else 0
            report.num_nodes = len(model.graph.node)
            
            # Metadata
            metadata = {prop.key: prop.value for prop in model.metadata_props}
            report.framework = metadata.get("author", "Unknown")
            report.framework_version = metadata.get("version", "Unknown")
            report.task = metadata.get("task", "Unknown")
            
            # Clases
            if "names" in metadata:
                import ast
                try:
                    names_dict = ast.literal_eval(metadata["names"])
                    report.classes = {int(k): str(v) for k, v in names_dict.items()}
                except Exception:
                    report.classes = {0: "object"}
            else:
                report.classes = {0: "object"}
                
            # Inputs
            for inp in model.graph.input:
                shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
                dtype = inp.type.tensor_type.elem_type
                report.inputs.append(TensorInfo(name=inp.name, shape=shape, dtype=str(dtype)))
                
            # Outputs
            for out in model.graph.output:
                shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
                dtype = out.type.tensor_type.elem_type
                report.outputs.append(TensorInfo(name=out.name, shape=shape, dtype=str(dtype)))
                
            # Fallback deduction for task if missing
            if report.task == "Unknown":
                if len(report.outputs) == 1:
                    report.task = "detect"
                elif len(report.outputs) == 2:
                    report.task = "segment"
                    
        except Exception as e:
            print(f"Error inspeccionando {filepath}: {e}")
            
        return report
