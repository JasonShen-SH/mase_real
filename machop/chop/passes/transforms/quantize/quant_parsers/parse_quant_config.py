from functools import partial

from .utils import cp_multi_values, has_multi_keys

""" QUANT_ARITH_ENTRIES
A mapping from (quantization arithmetic name) to (a mapping from (operand name) to (operand quantization spec name))

Example

A integer quantized value is defined by (width, frac_width), thus the mapping is defined as follows:
```python
"integer": {
    "weight_entries": ("weight_width", "weight_frac_width"),
    "data_in_entries": ("data_in_width", "data_in_frac_width"),
    "bias_entries": ("bias_width", "bias_frac_width"),
},
```
"""
QUANT_ARITH_ENTRIES = {
    # <arith_name> : {<operand_name> : (<operand_quantization_spec_name>,)}
    "integer": {
        "weight_entries": ("weight_width", "weight_frac_width"),
        "data_in_entries": ("data_in_width", "data_in_frac_width"),
        "bias_entries": ("bias_width", "bias_frac_width"),
    },
    "binary": {
        "weight_entries": (
            "weight_width",
            "weight_stochastic",
            "weight_bipolar",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_stochastic",
            "data_in_bipolar",
        ),
        "bias_entries": (
            "bias_width",
            "bias_stochastic",
            "bias_bipolar",
        ),
    },
    "ternary": {
        "weight_entries": (
            "weight_width",
            "weight_scaling_factor",
            "weight_mean",
            "weight_median",
            "weight_max",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_scaling_factor",
            "data_in_mean",
            "data_in_median",
            "data_in_max",
        ),
        "bias_entries": (
            "bias_width",
            "bias_scaling_factor",
            "bias_mean",
            "bias_max",
            "bias_median",
        ),
    },
    "minifloat_ieee": {
        "weight_entries": (
            "weight_width",
            "weight_exponent_width",
            "weight_exponent_bias",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_exponent_width",
            "data_in_exponent_bias",
        ),
        "bias_entries": ("bias_width", "bias_exponent_width", "bias_exponent_bias"),
    },
    "minifloat_denorm": {
        "weight_entries": (
            "weight_width",
            "weight_exponent_width",
            "weight_exponent_bias",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_exponent_width",
            "data_in_exponent_bias",
        ),
        "bias_entries": ("bias_width", "bias_exponent_width", "bias_exponent_bias"),
    },
    "log": {
        "weight_entries": ("weight_width", "weight_exponent_bias"),
        "data_in_entries": ("data_in_width", "data_in_exponent_bias"),
        "bias_entries": ("bias_width", "bias_exponent_bias"),
    },
    "block_fp": {
        "weight_entries": (
            "weight_width",
            "weight_exponent_width",
            "weight_exponent_bias",
            "weight_block_size",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_exponent_width",
            "data_in_exponent_bias",
            "data_in_block_size",
        ),
        "bias_entries": (
            "bias_width",
            "bias_exponent_width",
            "bias_exponent_bias",
            "bias_block_size",
        ),
    },
    "block_minifloat": {
        "weight_entries": (
            "weight_width",
            "weight_exponent_width",
            "weight_exponent_bias_width",
            "weight_block_size",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_exponent_width",
            "data_in_exponent_bias_width",
            "data_in_block_size",
        ),
        "bias_entries": (
            "bias_width",
            "bias_exponent_width",
            "bias_exponent_bias_width",
            "bias_block_size",
        ),
    },
    "block_log": {
        "weight_entries": (
            "weight_width",
            "weight_exponent_bias_width",
            "weight_block_size",
        ),
        "data_in_entries": (
            "data_in_width",
            "data_in_exponent_bias_width",
            "data_in_block_size",
        ),
        "bias_entries": (
            "bias_width",
            "bias_exponent_bias_width",
            "bias_block_size",
        ),
    },
}


""" cp_<entry_name> functions
A collection of functions to copy values from a src config to a parsed config.
"""


def cp_name(config: dict, p_config: dict, entries=None):
    cp_multi_values(config, p_config, ("name",))


def cp_bypass(config: dict, p_config: dict, entries=None):
    cp_multi_values(config, p_config, ("bypass",))


def cp_weight_entries(config: dict, p_config: dict, entries: dict):
    cp_multi_values(config, p_config, entries["weight_entries"])


def cp_data_in_entries(config: dict, p_config: dict, entries: dict):
    cp_multi_values(config, p_config, entries["data_in_entries"])


def cp_bias_entries(config: dict, p_config: dict, entries: dict):
    cp_multi_values(config, p_config, entries["bias_entries"])


def cp_weight_entries_to_bias(config: dict, p_config: dict, entries: dict):
    if has_multi_keys(config, entries["bias_entries"]):
        cp_multi_values(config, p_config, entries["bias_entries"])
    else:
        cp_multi_values(
            config, p_config, entries["weight_entries"], entries["bias_entries"]
        )


"""QUANT_ARITH_TO_CP_FN
a map from quant_arith to a collection of functions where each function copies a specific quant_arith_spec from a src config to a parsed config.

<quant_arith>: {
   "name": cp_name_function_<quant_arith>,
   "weight_entries": cp_weight_entries_function_<quant_arith>,
   "data_in_entries": cp_data_in_entries_function_<quant_arith>,
   "bias_entries": cp_bias_entries_function_<quant_arith>,
   "weight_entries_to_bias": cp_weight_entries_to_bias_function_<quant_arith>
}
"""
QUANT_ARITH_TO_CP_FN = {}


for quant_arith, entries in QUANT_ARITH_ENTRIES.items():
    QUANT_ARITH_TO_CP_FN[quant_arith] = {
        "name": partial(cp_name, entries=entries),
        "bypass": partial(cp_bypass, entries=entries),
        "weight_entries": partial(cp_weight_entries, entries=entries),
        "data_in_entries": partial(cp_data_in_entries, entries=entries),
        "bias_entries": partial(cp_bias_entries, entries=entries),
        "weight_entries_to_bias": partial(cp_weight_entries_to_bias, entries=entries),
    }

""" MASE_OP_TO_ENTRIES
a map from mase_op to a collection of required and optional entries.
"""
MASE_OP_TO_ENTRIES = {
    # <op_name> : (<required_entries>, <optional_entries>)
    "add": (("name", "data_in_entries"), ("bypass",)),
    "bmm": (("name", "data_in_entries", "weight_entries"), ("bypass",)),
    "conv1d": (
        ("name", "data_in_entries", "weight_entries"),
        ("bias_entries", "bypass"),
    ),
    "conv2d": (
        ("name", "data_in_entries", "weight_entries"),
        ("bias_entries", "bypass"),
    ),
    "matmul": (("name", "data_in_entries", "weight_entries"), ("bypass",)),
    "mul": (("name", "data_in_entries"), ("bypass",)),
    "linear": (
        ("name", "data_in_entries", "weight_entries"),
        ("bias_entries", "bypass"),
    ),
    "relu": (("name", "data_in_entries"), ("bypass",)),
    "sub": (("name", "data_in_entries"), ("bypass",)),
}


def optional_operand_entry_exists(config: dict, entry_name: str) -> bool:
    entry_name = entry_name.removesuffix("_entries")
    for key in config.keys():
        if key.startswith(entry_name):
            return True
    return False


def parse_node_config(config: dict, mase_op: str) -> dict:
    assert mase_op in MASE_OP_TO_ENTRIES, f"Unknown mase op: {mase_op}"
    if config.get("bypass", False):
        return config
    op_entries, op_optional_entries = MASE_OP_TO_ENTRIES[mase_op]
    assert isinstance(op_entries, tuple), f"op_entries must be a tuple: {op_entries}"
    assert isinstance(
        op_optional_entries, tuple
    ), f"op_optional_entries must be a tuple: {op_optional_entries}"
    p_config = {}
    for entry in op_entries:
        entry_cp_fn = QUANT_ARITH_TO_CP_FN[config["name"]][entry]
        entry_cp_fn(config, p_config)
    for entry in op_optional_entries:
        if optional_operand_entry_exists(config, entry):
            entry_cp_fn = QUANT_ARITH_TO_CP_FN[config["name"]][entry]
            entry_cp_fn(config, p_config)
    return p_config
