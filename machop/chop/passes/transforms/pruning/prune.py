"""
The entrypoint to chop's pruning transforms

The pruning passes are public-facing entrypoints to the functionality exposed by the 
pruning transform. Currently, we support a layerwise pruning strategy for weights and
support activation pruning as an add-on. Please take a look at the methods.py file for
more details on the pruning methods. The iterator steps through compatible nodes in the
graph, wraps them with an activation pruning handler (if applicable), and prunes them.

NOTE: We only support vision models targeted for classification tasks. Also, we
currently only support one-shot, locally-scoped unstructured pruning; we don't support
fine-tuning with pruning yet.

Example Usage:
# For example configurations, see the configs/tests/prune directory
$ python tests/passes/transforms/prune/toy.py
$ python tests/passes/transforms/prune/prune.py <config_name>
(or)
$ ./ch transform --config configs/tests/prune/resnet18_fixed_global.toml --pretrained

----------------------------------------------------------------------------------------
Internal Backlog:
1. Add support for fine-tuning with pruning via registering weight masks via the wrap
   function of the pruner. The unwrap pass would then take the responsibility of
   removing these masks to make the pruning permanent. The exact specifics may require
   a bit of thought as the model will have to be saved and then loaded again for
   training. Perhaps we could create a wrapper class (nn.Module) that uses a custom
   forward function to apply the masks while training? 
2. Currently, our calibration pass only uses a single batch to generate a report on the
   channel-wise activation sparsity. Perhaps we could aggregate statistics over multiple
   batches to get a better estimate.
3. Add tests for the weight and activation configuration validation functions.
4. Consider extending PyTorch's built-in pruning module to support activation sparsity;
   this is an attractive option as it would allow us to leverage the existing pruning
   workflows (incl. globally unstructured ones).
5. Provide user control over exactly what is logged via the handler. That is, config.
   params. for log_input, log_values, log_statistics, log_summary, log_thresholds, etc.
"""

from functools import partial

import torch
import torch.nn as nn
from tqdm.contrib.logging import tqdm_logging_redirect

from chop.passes.graph.mase_graph import MaseGraph
from chop.passes.utils import get_mase_op, get_mase_type, get_node_actual_target
from chop.tools.logger import getLogger

from .criteria import RANK_CRITERIA
from .methods import (
    LevelPruner,
    ChannelPruner,
    ACTIVATION_PRUNE_STRATEGIES,
    PRUNE_SCOPES,
    ActivationPruneHandler,
)
from .utilities import (
    measure_sparsity,
    count_parameters,
    estimate_model_size,
    count_buffers,
)

# Constants ----------------------------------------------------------------------------
# Pruneable MASE operators
# NOTE: We don't do activation pruning for conv1d and linear layers.
PRUNEABLE_OPS = {"conv1d": nn.Conv1d, "conv2d": nn.Conv2d, "linear": nn.Linear}

# A registry of available pruning strategies (i.e. algorithms)
PRUNE_METHODS = {
    # A basic one-shot pruner that prunes to a given sparsity level
    "level-pruner": LevelPruner,
    "channel-pruner": ChannelPruner,
    # Add more here...
}

# Housekeeping -------------------------------------------------------------------------
logger = getLogger(__file__)
logger.propagate = False  # Avoid duplicate logs


# Passes -------------------------------------------------------------------------------
# NOTE: This is where you'd handle layer-specific pruning (i.e. by name, type, etc.)
# by calling different graph iterators. For now, we stick with a simple iterator
# that prunes all supported operators.
# See machop/chop/passes/transforms/quantize/quantize.py for an example.
def prune_transform_pass(graph: MaseGraph, save_dir=None, config=None):
    return prune_graph_iterator(graph, save_dir, config)


# NOTE: For now, this pass takes care of unwrapping the activation prune handler, but
# later this will be useful for making pruning permanent by removing masks.
def prune_unwrap_transform_pass(graph: MaseGraph, *_):
    # Remove the activation hooks and the handler from the model
    handler = getattr(graph.model, "activation_prune_handler", None)
    if handler:
        handler.unwrap()
        delattr(graph.model, "activation_prune_handler")
        logger.info("Removed activation pruning hooks and handler from the model")
    else:
        logger.info(
            "No activation handler found in the model. Did you make sure to load the "
            "model from a mase graph module (.mz) file? Skipping unwrap."
        )
    return graph


# Iterators ----------------------------------------------------------------------------
def prune_graph_iterator(graph: MaseGraph, save_dir: str, config: dict):
    # This generator is used to create a batch for a sample forward pass.
    input_generator = config.get("input_generator", None)

    # Setup all pruning-related parameters (incl. basic validation)
    method, iterate, kwargs = get_weight_prune_params(config["weight"], graph)

    prune_activations = "activation" in config
    strategy, target = (
        get_activation_prune_params(config["activation"], graph)
        if prune_activations
        else (None, None)
    )

    # Create and register the handler as an attribute of the model; since we're using
    # partial functions for the hooks and the handler is passed in as an argument, we'd
    # lose the reference when saving the graph if we didn't do this.
    handler = None
    if prune_activations:
        handler = ActivationPruneHandler(target, strategy, save_dir)
        # NOTE: DO NOT enable this unless you have a LOT of disk space. This will log
        # the sparsity of nearly every single patch processed; handler.samples can help
        # but it'll still hog memory. See handler._conv_forward method for more details.
        handler.log_values = False
        setattr(graph.model, "activation_prune_handler", handler)
        logger.debug("Added the activation prune handler as a model attribute")
    pruner = method(**kwargs)

    # Log metadata about the graph before pruning
    _log_metadata(graph)

    # Iterate over the graph and prune the compatible nodes; note that we do two passes.
    if iterate:
        _graph_iterator(graph, partial(_wrap_callback, pruner, handler))
        _graph_iterator(graph, partial(_apply_callback, pruner))
    else:
        # This is a special case for the channel pruner. We just pass the graph model
        # to the pruner and it takes care of the rest.
        graph = pruner.prune(graph.model, input_generator)

    # A sample forward pass to log activation sparsity statistics :)
    # NOTE: Currently, we only do this on one batch sample. As in line 230 of
    # machop/chop/passes/analysis/statistical_profiler/profile_statistics.py,
    # we may want to aggregate statistics over multiple batches.
    if handler:
        logger.info("Running inference to generate an activation sparsity report")
        previous_state = graph.model.training
        graph.model.train(False)
        with torch.inference_mode():
            batch = next(input_generator)["x"]
            graph.model(batch)
        graph.model.train(previous_state)

    # Summary
    logger.info("Weight Pruning Summary:")
    pruner.show_summary()
    if prune_activations:
        logger.info("Activation Pruning Summary:")
        handler.show_summary()

    # Log metadata about the graph before pruning
    _log_metadata(graph)

    # Save the pruning report and summary
    pruner.save_summary(save_dir)
    if handler:
        handler.save_statistics(save_dir)
        handler.save_summary(save_dir)

    # Set the handler logging attributes to False for better performance. Also, we
    # remove the handler's forward hook that's responsible for collecting statistics.
    if handler:
        handler.unwrap(keep_pre=True)
        handler.clear_statistics()  # Helps make the saved graph smaller

        # NOTE: The verify, log_input, log_values, and log_statistics toggles have no
        # purpose after unwrapping; they're associated with the removed forward hook.
        handler.verify = False
        handler.log_input = False
        handler.log_values = False
        handler.log_summary = False
        handler.log_statistics = False
        handler.log_thresholds = False

    return graph


# Helper functions ---------------------------------------------------------------------
def _graph_iterator(graph: MaseGraph, callback: callable):
    total = len(graph.fx_graph.nodes)
    with tqdm_logging_redirect(total=total, loggers=[logger]) as pbar:
        for node in graph.fx_graph.nodes:
            if get_mase_op(node) in ["batch_norm1d", "batch_norm2d"]:
                logger.warning(
                    f"Batchnorm layer detected ({node.target}). This could corrupt "
                    "sparsity in the weights when fused later."
                )

            if get_mase_op(node) not in PRUNEABLE_OPS.keys():
                # Skip if the operator is not supported for pruning
                pbar.update(1)
                continue

            if get_mase_type(node) == "module_related_func":
                callback(node, pbar)

            pbar.update(1)
        pbar.set_description("Done")


def _wrap_callback(pruner, handler, node, pbar):
    module = get_node_actual_target(node)

    # Wrap and prune the module :)
    pbar.set_description(f"Wrapping candidate node {node.target}")
    pruner.wrap(module, node.target)
    if handler:
        handler.wrap(module, node.target)


def _apply_callback(pruner, node, pbar):
    module = get_node_actual_target(node)

    # NOTE: This is where you'd prepare keyword args for a custom criterion
    kwargs = {}

    pbar.set_description(f"Pruning candidate node {node.target}")
    pruner.apply(module, node.target, **kwargs)


# TODO: We need to add tests for these functions with a variety of configurations.
def get_weight_prune_params(config: dict, graph: MaseGraph):
    method = config.get("method", "level-pruner")
    criterion = config.get("criterion", "random")
    sparsity = config.get("sparsity", 0.0)
    scope = config.get("scope", "local")

    # Validate the parameters
    if method not in PRUNE_METHODS.keys():
        raise ValueError(
            "Unsupported pruning method {}. Please choose from {}".format(
                method, list(PRUNE_METHODS.keys())
            )
        )
    if criterion not in RANK_CRITERIA.keys():
        raise ValueError(
            "Unsupported ranking criterion {}. Please choose from {}".format(
                criterion, list(RANK_CRITERIA.keys())
            )
        )
    if scope not in PRUNE_SCOPES:
        raise ValueError(
            "Unsupported pruning scope {}. Please choose from {}".format(
                scope, PRUNE_SCOPES
            )
        )
    if isinstance(sparsity, dict):
        # Make sure that the scope is local
        if scope != PRUNE_SCOPES[0]:  # not local
            raise ValueError("Layer-wise budgets only possible with a local scope!")

        # Verify that the keys are valid node names and that the values are valid
        names = [node.target for node in graph.fx_graph.nodes]
        for name in sparsity.keys():
            if name not in names:
                raise ValueError(f"Node {name} not found in the graph!")
            _verify_sparsity(sparsity[name])
    else:
        _verify_sparsity(sparsity)

    iterate = not (method == "channel-pruner")
    criterion = RANK_CRITERIA[criterion]
    kwargs = {"criterion": criterion, "sparsity": sparsity, "scope": scope}
    if method == "channel-pruner":
        # NOTE: In this case, the criterion, sparsity and scope are all ignored. We
        # only use Microsoft NNI's L1NormPruner for now.
        kwargs.clear()
        kwargs["config_list"] = config.get("config_list", [{"sparsity_per_layer": 0.5}])

    method = PRUNE_METHODS[method]
    return method, iterate, kwargs


def get_activation_prune_params(config: dict, graph: MaseGraph):
    strategy = config.get("strategy", "fixed-global")
    target = config.get("target", 0.1)

    # Validate the parameters
    if strategy not in ACTIVATION_PRUNE_STRATEGIES:
        raise ValueError(
            "Unsupported activation pruning strategy {}. Please choose from {}".format(
                strategy, list(ACTIVATION_PRUNE_STRATEGIES)
            )
        )
    # Validate the target based on the choice of strategy
    # NOTE: In the observe strategy, the target isn't used, so we don't care about it.
    if strategy == ACTIVATION_PRUNE_STRATEGIES[3]:  # fixed-layerwise
        # Verify that the target's a dictionary and that its keys are valid node names
        if not isinstance(target, dict):
            raise ValueError(
                "Expected dictionary of layerwise thresholds. Got {}".format(
                    type(target)
                )
            )
        names = [node.target for node in graph.fx_graph.nodes]  # Names of all nodes
        for name in target.keys():
            if name not in names:
                raise ValueError(f"Node {name} not found in the graph!")
    else:
        # Verify that the target's a float
        if not isinstance(target, float):
            raise ValueError("Target must be a float. Got {}".format(type(target)))
        # Verify that the target's between 0 and 1 if the strategy is adaptive
        # NOTE: Here, the target is treated like sparsity whereas in both the other
        # cases, it's treated like a threshold.
        if strategy == ACTIVATION_PRUNE_STRATEGIES[1] and (target < 0 or target > 1):
            raise ValueError("Target must be between 0 and 1. Got {}".format(target))

    return strategy, target


def _verify_sparsity(sparsity):
    if not isinstance(sparsity, float):
        raise ValueError("Sparsity must be a float. Got {}".format(type(sparsity)))
    if sparsity < 0 or sparsity > 1:
        raise ValueError("Sparsity must be between 0 and 1. Got {}".format(sparsity))


def _log_metadata(graph: MaseGraph):
    logger.info(
        "\n"
        f"Sparsity    : {measure_sparsity(graph.model):>14.3f}\n"
        f"Params (TOT): {count_parameters(graph.model):>14,}\n"
        f"Params (NNZ): {count_parameters(graph.model, nonzero_only=True):>14,}\n"
        f"Buffers     : {count_buffers(graph.model):>14,}\n"
        f"Size        : {estimate_model_size(graph.model):>14.3f} MB"
    )
