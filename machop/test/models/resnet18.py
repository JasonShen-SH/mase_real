#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import os
import sys

import torch
import torch.nn as nn

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "machop"
    )
)


from chop.dataset import get_dataset_info
from chop.models import get_model
from chop.passes.analysis.report import report_graph_analysis_pass
from chop.passes.analysis.verify import verify
from chop.passes.graph.mase_graph import MaseGraph


# --------------------------------------------------
#   Emit Verilog using Mase
# --------------------------------------------------
def main():
    load_pretrained = True
    cifar10_info = get_dataset_info("cifar10")
    resnet18 = get_model(
        "resnet18", task="cls", dataset_info=cifar10_info, pretrained=load_pretrained
    )
    mg = MaseGraph(model=resnet18)
    # print(mg.fx_graph)

    # Sanity check and report
    # mg = quantize(mg)
    # mg = verify(mg)
    # mg = report(mg)
    # mg = emit_verilog(mg)


# --------------------------------------------------
#   Execution
# --------------------------------------------------
main()
