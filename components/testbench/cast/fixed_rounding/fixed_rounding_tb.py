#!/usr/bin/env python3

# This script tests the fixed point castiplier
import random, os, logging
import sys
import cocotb
from cocotb.triggers import Timer
from cocotb.runner import get_runner

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append("/workspace/machop/")
import torch
import torch.nn as nn
from z_qlayers import _integer_quantize
from z_qlayers import quantize_to_int as q2i

debug = False

logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)


class VerificationCase:
    def __init__(self, samples=2):
        self.in_width = 16
        self.in_frac_width = 6
        self.out_width = 8
        self.out_frac_width = 5
        self.in_size = 3
        self.inputs, self.outputs = [], []
        torch.manual_seed(0)
        for _ in range(samples):
            i, o = self.single_run()
            self.inputs.append(i)
            self.outputs.append(o)
        self.samples = samples

    def single_run(self):
        data = _integer_quantize(
            torch.tensor([2.625, 2.625, 2.625]), self.in_width, self.in_frac_width
        )
        print("data = ", data)
        h = q2i(data, self.in_width, self.in_frac_width)
        print("h = ", h)
        print("s = ", self.sw_cast(data))
        return [int(i) for i in h], [int(i) for i in self.sw_cast(data)]

    def sw_cast(self, inputs):
        outputs = q2i(inputs, self.out_width, self.out_frac_width)
        return outputs

    def get_dut_parameters(self):
        return {
            "IN_WIDTH": self.in_width,
            "IN_FRAC_WIDTH": self.in_frac_width,
            "OUT_WIDTH": self.out_width,
            "OUT_FRAC_WIDTH": self.out_frac_width,
        }

    def get_dut_input(self, i):
        return self.inputs[i]

    def get_dut_output(self, i):
        return self.outputs[i]


def check_outputs(hw_out, sw_out):
    assert len(hw_out) == len(
        sw_out
    ), "Mismatched output size: {} expected = {}".format(len(hw_out), len(sw_out))
    for i in range(len(hw_out)):
        assert (
            hw_out[i] == sw_out[i]
        ), "Mismatched output value {}: {} expected = {}".format(
            i, int(hw_out[i]), sw_out[i]
        )
    return True


@cocotb.test()
async def test_fixed_rounding(dut):
    """Test random number casting"""
    test_case = VerificationCase(samples=20)

    # set inputs outputs
    for i in range(test_case.samples):
        x = test_case.get_dut_input(i)
        y = test_case.get_dut_output(i)
        dut.data_in.value = x
        await Timer(2, units="ns")
        assert check_outputs(
            [int(v) for v in dut.data_out.value], y
        ), "Output are incorrect on the {}th cycle: {} (expected {} => {})".format(
            i, [int(v) for v in dut.data_out.value], x, y
        )


def runner():
    sim = os.getenv("SIM", "verilator")
    verilog_sources = ["../../../cast/fixed_rounding.sv"]

    test_case = VerificationCase()

    # set parameters
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f"-G{k}={v}")
    print(extra_args)

    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="fixed_rounding",
        build_args=extra_args,
    )
    runner.test(hdl_toplevel="fixed_rounding", test_module="fixed_rounding_tb")


if __name__ == "__main__":
    runner()
