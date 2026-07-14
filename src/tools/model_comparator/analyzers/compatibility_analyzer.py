import json
from tools.model_comparator.core.report_model import ModelReport

class CompatibilityAnalyzer:
    """Valida la compatibilidad de un modelo ONNX contra un perfil esperado."""
    
    @staticmethod
    def analyze(report: ModelReport, expected_profile: dict) -> dict:
        result = {
            "model": report.model_name,
            "status": "COMPATIBLE",
            "checks": {},
            "reason": "",
            "required_changes": []
        }
        
        # Check inputs
        exp_input = expected_profile.get("input_shape", [])
        if report.inputs and len(report.inputs) > 0:
            found_input = report.inputs[0].shape
            match = (exp_input == found_input)
            result["checks"]["inputs"] = {"expected": str(exp_input), "found": str(found_input), "match": match}
            if not match:
                result["status"] = "NOT_COMPATIBLE"
                result["reason"] += "Input shape mismatch. "
        
        # Check output0
        exp_output = expected_profile.get("output_shape", [])
        if report.outputs and len(report.outputs) > 0:
            found_output = report.outputs[0].shape
            match = (exp_output == found_output)
            result["checks"]["outputs"] = {"expected": str(exp_output), "found": str(found_output), "match": match}
            if not match:
                result["status"] = "NOT_COMPATIBLE"
                result["reason"] += f"La Plataforma espera stride={exp_output[-1] if exp_output else 'N/A'} y el modelo exporta stride={found_output[-1]}. "
                result["required_changes"].append("modificar parseo de tensor")
                
        # Check number of outputs
        exp_num_outputs = expected_profile.get("num_outputs", 1)
        found_num_outputs = len(report.outputs)
        if exp_num_outputs != found_num_outputs:
            result["status"] = "NOT_COMPATIBLE"
            result["reason"] += f"Expected {exp_num_outputs} outputs, found {found_num_outputs}. "
            if found_num_outputs == 2:
                result["required_changes"].extend(["leer output1", "decodificar máscaras", "generar polígonos"])
                
        if result["status"] == "COMPATIBLE":
            result["reason"] = "Model perfectly matches the expected platform profile."
            
        return result
