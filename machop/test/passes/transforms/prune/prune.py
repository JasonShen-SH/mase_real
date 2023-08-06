#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog


import logging
import os
import sys
from pathlib import Path

import toml

# Housekeeping -------------------------------------------------------------------------
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(Path(__file__).resolve().parents[4].as_posix())

from chop.passes import (
    add_mase_ops_analysis_pass,
    init_metadata_analysis_pass,
    prune_transform_pass,
)
from chop.models.toy import ToyConvNet
from chop.passes.graph.mase_graph import MaseGraph
from chop.tools.logger import getLogger


logger = getLogger("chop")
logger.setLevel(logging.DEBUG)


def main():
    model = ToyConvNet(num_classes=10)
    graph = MaseGraph(model=model)

    # NOTE: Both functions have pass arguments that are not used in this example
    graph = init_metadata_analysis_pass(graph, None)
    graph = add_mase_ops_analysis_pass(graph, None)
    logger.debug(graph.fx_graph)

    root = Path(__file__).resolve().parents[4]
    config_path = root / "configs/tests/prune/simple_unstructured.toml"
    with open(config_path, "r") as f:
        config = toml.load(f)
        config = config["passes"]["prune"]

        graph = prune_transform_pass(graph, config)


if __name__ == "__main__":
    main()
