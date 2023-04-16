"""
hook function for recoding tensor shapes during forward propagation

add metadata["common"]["args"/"results"]["data_in"/"weight"/"bias"/"data_out"]["size"]
"""
import operator
import traceback
from collections import defaultdict
from logging import getLogger
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.node import Node, map_aggregate
from torchvision.ops.stochastic_depth import stochastic_depth

from ....modify.quantizers.functions import (
    add_integer,
    bmm_integer,
    matmul_integer,
    relu_integer,
)
from ....modify.quantizers.layers import AddInteger

logger = getLogger(__name__)


def _torch_size_to_tuple(tensor_shape):
    return tuple(shape_i for shape_i in tensor_shape)


def _get_packed_shape(tensor_list):
    shape = []
    for tensor_i in tensor_list:
        assert isinstance(tensor_i, torch.Tensor)
        shape.append(_torch_size_to_tuple(tensor_i.shape))


def _get_tuple_shape_and_is_packed(x):
    if isinstance(x, torch.Tensor):
        return _torch_size_to_tuple(x.shape), False
    elif isinstance(x, (int, float)):
        return (1,), False
    elif isinstance(x, torch.Size):
        return (len(x),), False
    elif isinstance(x, (list, tuple)):
        if isinstance(x[0], torch.Tensor):
            return _get_packed_shape(x), True
        else:
            raise RuntimeError
    else:
        raise RuntimeError


def _set_arg_size_before_call_function(node: Node, function, args, kwargs):
    meta_common_args = node.meta["common"]["args"]
    if function in (
        F.relu,
        relu_integer,
        F.hardsigmoid,
        F.hardswish,
        F.silu,
        F.softmax,
        operator.add,
        torch.add,
        add_integer,
        operator.mul,
        torch.mul,
        operator.floordiv,
        torch.floor_divide,
        operator.eq,
        torch.eq,
        torch.unbind,
        torch.mean,
        torch.matmul,
        torch.bmm,
        matmul_integer,
        bmm_integer,
        torch.reshape,
        torch.flatten,
        torch.transpose,
        torch.permute,
        F.dropout,
        F.dropout1d,
        F.dropout2d,
        F.dropout3d,
        operator.getitem,
        stochastic_depth,
        torch._assert,
        getattr,
    ):
        if len(node.all_input_nodes) == 1:
            (
                meta_common_args["data_in"]["size"],
                meta_common_args["data_in"]["is_packed"],
            ) = _get_tuple_shape_and_is_packed(args[0])
            meta_common_args["data_in"].pop("from")
        else:
            for i in range(len(node.all_input_nodes)):
                (
                    meta_common_args[f"data_in_{i}"]["size"],
                    meta_common_args[f"data_in_{i}"]["is_packed"],
                ) = _get_tuple_shape_and_is_packed(args[i])
                meta_common_args[f"data_in_{i}"]["from"] = node.all_input_nodes[i]
    elif function in (torch.cat, torch.concat):
        for i in range(len(node.all_input_nodes)):
            (
                meta_common_args[f"data_in_{i}"]["size"],
                meta_common_args[f"data_in_{i}"]["is_packed"],
            ) = _get_tuple_shape_and_is_packed(args[0][i])
            meta_common_args[f"data_in_{i}"]["from"] = node.all_input_nodes[i]
    else:
        logger.warning(f"Unrecognized function `{function}` when setting size")


def _set_result_size_after_call_function(node: Node, function, output):
    meta_common_results = node.meta["common"]["results"]

    if function in (
        F.relu,
        relu_integer,
        F.hardsigmoid,
        F.hardswish,
        F.silu,
        F.softmax,
        operator.add,
        torch.add,
        add_integer,
        operator.mul,
        torch.mul,
        operator.floordiv,
        torch.floor_divide,
        operator.eq,
        torch.eq,
        torch.concat,
        torch.cat,
        torch.unbind,
        torch.mean,
        torch.matmul,
        torch.bmm,
        matmul_integer,
        bmm_integer,
        torch.reshape,
        torch.flatten,
        torch.transpose,
        torch.permute,
        F.dropout,
        F.dropout1d,
        F.dropout2d,
        F.dropout3d,
        getattr,
        operator.getitem,
        stochastic_depth,
        torch._assert,
    ):
        if output is None:
            return
        else:
            (
                meta_common_results["data_out"]["size"],
                meta_common_results["data_out"]["is_packed"],
            ) = _get_tuple_shape_and_is_packed(output)
    else:
        logger.warning(f"Unrecognized function `{function}` when setting size")


def _set_arg_size_before_call_module(node: Node, module, args, kwargs):
    meta_common_args = node.meta["common"]["args"]
    meta_common_args["data_in"].pop("from")
    if isinstance(
        module, (nn.ReLU, nn.Hardswish, nn.Hardsigmoid, nn.SiLU, nn.Sigmoid, nn.GELU)
    ):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)

        # meta_common_args["data_in"].pop("from")
        # meta_common_args["data_in"]["from"] = node.all_input_nodes[0]
    elif isinstance(module, (nn.Softmax,)):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)
    elif isinstance(module, (nn.Embedding,)):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)
        meta_common_args["weight"]["size"] = _torch_size_to_tuple(module.weight.shape)
    elif isinstance(module, (AddInteger,)):
        meta_common_args["data_in_0"]["size"] = _torch_size_to_tuple(args[0].shape)
        meta_common_args["dat_in_0"]["from"] = node.all_input_nodes[0]
        meta_common_args["data_in_1"]["size"] = _torch_size_to_tuple(args[1].shape)
        meta_common_args["data_in_1"]["from"] = node.all_input_nodes[1]
    elif isinstance(module, (nn.Linear,)):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)
        meta_common_args["weight"]["size"] = _torch_size_to_tuple(module.weight.shape)
        if module.bias is not None:
            meta_common_args["bias"]["size"] = _torch_size_to_tuple(module.bias.shape)
    elif isinstance(module, (nn.Conv1d,)):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)
        meta_common_args["weight"]["size"] = _torch_size_to_tuple(module.weight.shape)
        if module.bias is not None:
            meta_common_args["bias"]["size"] = _torch_size_to_tuple(module.bias.shape)
    elif isinstance(module, (nn.Conv2d,)):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)
        meta_common_args["weight"]["size"] = _torch_size_to_tuple(module.weight.shape)
        if module.bias is not None:
            meta_common_args["bias"]["size"] = _torch_size_to_tuple(module.bias.shape)
    elif isinstance(module, (nn.BatchNorm2d,)):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)
        meta_common_args["weight"]["size"] = _torch_size_to_tuple(module.weight.shape)
        meta_common_args["bias"]["size"] = _torch_size_to_tuple(module.bias.shape)
        meta_common_args["running_mean"]["size"] = _torch_size_to_tuple(
            module.running_mean.shape
        )
        meta_common_args["running_var"]["size"] = _torch_size_to_tuple(
            module.running_var.shape
        )
    elif isinstance(module, (nn.LayerNorm,)):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)
        meta_common_args["weight"]["size"] = _torch_size_to_tuple(module.weight.shape)
        meta_common_args["bias"]["size"] = _torch_size_to_tuple(module.bias.shape)
    elif isinstance(
        module,
        (
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.MaxPool3d,
            nn.AdaptiveMaxPool1d,
            nn.AdaptiveMaxPool2d,
            nn.AdaptiveMaxPool3d,
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.AvgPool3d,
            nn.AdaptiveAvgPool1d,
            nn.AdaptiveAvgPool2d,
            nn.AdaptiveAvgPool3d,
        ),
    ):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)
    elif isinstance(
        module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.Identity)
    ):
        meta_common_args["data_in"]["size"] = _torch_size_to_tuple(args[0].shape)
    else:
        logger.warning(
            f"Unrecognized module `{type(module).__name__}` when setting size"
        )


def _set_result_size_after_call_module(node: Node, module, output):
    meta_common_results = node.meta["common"]["results"]
    if isinstance(
        module, (nn.ReLU, nn.Hardswish, nn.Hardsigmoid, nn.SiLU, nn.Sigmoid, nn.GELU)
    ):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(module, (nn.Softmax,)):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(module, (nn.Embedding,)):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(module, (AddInteger,)):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(module, (nn.Linear,)):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(module, (nn.Conv1d,)):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(module, (nn.Conv2d,)):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(module, (nn.BatchNorm2d,)):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(module, (nn.LayerNorm,)):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(
        module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.Identity)
    ):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)
    elif isinstance(
        module,
        (
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.MaxPool3d,
            nn.AdaptiveMaxPool1d,
            nn.AdaptiveMaxPool2d,
            nn.AdaptiveMaxPool3d,
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.AvgPool3d,
            nn.AdaptiveAvgPool1d,
            nn.AdaptiveAvgPool2d,
            nn.AdaptiveAvgPool3d,
        ),
    ):
        meta_common_results["data_out"]["size"] = _torch_size_to_tuple(output.shape)

    else:
        logger.warning(
            f"Unrecognized module `{type(module).__name__}` when setting size"
        )


def _set_arg_size_before_call_method(node: Node, method_name: str, args, kwargs):
    """
    self_obj.method(self, data_in_1), where 'self' is data_in_0
    """
    meta_common_args = node.meta["common"]["args"]
    if method_name in (
        "relu",
        "softmax",
        "add",
        "matmul",
        "bmm",
        "mean",
        "view",
        "reshape",
        "transpose",
        "permute",
        "flatten",
        "expand",
        "unbind",
        "contiguous",
        "size",
    ):
        if len(node.all_input_nodes) == 1:
            (
                meta_common_args["data_in"]["size"],
                meta_common_args["data_in"]["is_packed"],
            ) = _get_tuple_shape_and_is_packed(args[0])
            meta_common_args["data_in"].pop("from")
        else:
            for i in range(len(node.all_input_nodes)):
                (
                    meta_common_args[f"data_in_{i}"]["size"],
                    meta_common_args[f"data_in_{i}"]["is_packed"],
                ) = _get_tuple_shape_and_is_packed(args[i])
                meta_common_args[f"data_in_{i}"]["from"] = node.all_input_nodes[i]
    else:
        logger.warning(f"Unrecognized method name `{method_name}` when setting size")


def _set_result_size_after_call_method(node: None, method_name: str, output):
    meta_common_results = node.meta["common"]["results"]
    if method_name in (
        "relu",
        "softmax",
        "add",
        "matmul",
        "bmm",
        "mean",
        "view",
        "reshape",
        "transpose",
        "permute",
        "flatten",
        "expand",
        "unbind",
        "contiguous",
        "size",
    ):
        (
            meta_common_results["data_out"]["size"],
            meta_common_results["data_out"]["is_packed"],
        ) = _get_tuple_shape_and_is_packed(output)
    else:
        logger.warning(f"Unrecognized method name `{method_name}` when setting size")
