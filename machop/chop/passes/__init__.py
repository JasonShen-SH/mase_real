from .analysis import add_common_metadata_analysis_pass, init_metadata_analysis_pass
from .transforms import quantize_summary_analysis_pass, quantize_transform_pass, prune_transform_pass

analysis_passes = ["init_metadata", "add_common_metadata"]
transform_passes = ["quantize"]

passes = {
    "init_metadata": init_metadata_analysis_pass,
    "add_common_metadata": add_common_metadata_analysis_pass,
    "quantize": quantize_transform_pass,
    "quantize_summary": quantize_summary_analysis_pass,
    'prune': prune_transform_pass,
}
