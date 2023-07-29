#! /usr/bin/env python3
# ---------------------------------------
# This script runs the regression model for hls
# ---------------------------------------

from argparse import ArgumentParser
import os

from regression_gen import int_linear2d_dse, int_softmax_dse, int_layernorm_dse


def run(args):
    op = args.op
    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    mode = args.mode
    top = args.dir
    if op == "int_linear2d":
        int_linear2d_dse(mode=mode, top=top)
    elif op == "int_softmax":
        int_softmax_dse(mode=mode, top=top)
    elif op == "int_layernorm":
        int_layernorm_dse(mode=mode, top=top)
    else:
        assert False, f"Unsupported op = {op}"


# ---------- main function --------------
def main():
    USAGE = """Usage:
mase_hls  ...
"""

    parser = ArgumentParser(usage=USAGE)
    parser.add_argument(
        "--op",
        dest="op",
        default=None,
        help="Op name to explore",
    )
    parser.add_argument(
        "--dir",
        dest="dir",
        default=os.path.join(os.getcwd(), "dse"),
        help="Directory to store files",
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        default=None,
        help="Mode to run: codegen, synth, report, all",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
