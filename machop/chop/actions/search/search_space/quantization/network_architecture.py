from copy import deepcopy
from torch import nn
from ..base import SearchSpaceBase
from .....passes.graph.transforms.quantize import (
    QUANTIZEABLE_OP,
    quantize_transform_pass,
)
from .....ir.graph.mase_graph import MaseGraph
from .....passes.graph import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
)
from .....passes.graph.utils import get_mase_op, get_mase_type
from ..utils import flatten_dict, unflatten_dict
from collections import defaultdict
from chop.passes.graph.utils import get_parent_name
import pdb
from chop.tools import load_model


DEFAULT_NETWORK_CONFIG = {
    "config": {
        "by":"name",
        "default": {"config": {"name": None}},
        "seq_blocks_2": {
            "config": {
                "name": "output_only",
                "channel_multiplier_output": 2,
                }
            },

        "seq_blocks_3": {
            "config": {
                "name": "relu",
                }
            },

        "seq_blocks_4": {
            "config": {
                "name": "both",
                "channel_multiplier_input": 2,
                "channel_multiplier_output": 2,
                }
            },

        "seq_blocks_5": {
            "config": {
                "name": "relu",
                }
            },

        "seq_blocks_6": {
            "config": {
                "name": "input_only",
                "channel_multiplier_input": 2,
                }
            },
    }   
}

class NetworkArchitectureSearch(SearchSpaceBase):
        
    def _post_init_setup(self):
        self.model.to("cpu") 
        self.mg = None 
        self._node_info = None 
        self.default_config = DEFAULT_NETWORK_CONFIG

        # change network architecture by layer name or layer type
        assert (
            "by" in self.config["setup"]
        ), "Must specify entry `by` (config['setup']['by] = 'name' or 'type')"


    def instantiate_linear(self, in_features, out_features, bias):
        if bias is not None:
            bias = True
        return nn.Linear(
            in_features=in_features, 
            out_features=out_features, 
            bias=bias)

    def redefine_linear_transform_pass(self, graph, pass_args=None):
        # graph = self.mg
        pass_args_copy = deepcopy(pass_args)
        main_config = pass_args_copy.pop('config')
        default = main_config.pop('default', None)
        if default is None:
            raise ValueError(f"default value must be provided.")
        i = 0
        for node in graph.fx_graph.nodes: 
            i += 1
            # if node name is not matched, it won't be tracked
            config = main_config.get(node.name, default)['config']
            name = config.get("name", None)
            if name != None and name != "relu":
                ori_module = graph.modules[node.target]
                in_features = ori_module.in_features
                out_features = ori_module.out_features
                bias = ori_module.bias
                if name == "output_only":
                    out_features = out_features * config["channel_multiplier_output"] 
                elif name == "both":
                    in_features = in_features * config["channel_multiplier_input"] 
                    out_features = out_features * config["channel_multiplier_output"] 
                elif name == "input_only":
                    in_features = in_features * config["channel_multiplier_input"]  
                new_module = self.instantiate_linear(in_features, out_features, bias)
                parent_name, name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)
        return graph, {}
    
    def instantiate_relu(self, boolean):
        return nn.ReLU(inplace=boolean)

    def redefine_relu_pass(self, graph, pass_args=None):
        # graph = self.mg
        pass_args_copy = deepcopy(pass_args)
        main_config = pass_args_copy.pop('config')
        default = main_config.pop('default', None)
        if default is None:
            raise ValueError(f"default value must be provided.")
        i = 0
        for node in graph.fx_graph.nodes:
            i += 1
            config = main_config.get(node.name, default)['config']
            name = config.get("name", None)
            if name == "relu":
                new_module = self.instantiate_relu(True)
                parent_name, name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)
        return graph, {}


    def rebuild_model(self, sampled_config, is_eval_mode: bool = True):
        # set train/eval mode before creating mase graph
        if is_eval_mode:
            self.model.eval()
        else:
            self.model.train()

        # init masegraph

        '''
        if self.mg is None:
            assert self.model_info.is_fx_traceable, "Model must be fx traceable"
            mg = MaseGraph(self.model)
            mg,_ = init_metadata_analysis_pass(mg, None)
            self.mg = mg
        else:
            mg = self.mg
        '''
        numbers = [
            sampled_config['config']['seq_blocks_2']['config'].get('channel_multiplier_output', 1),
            sampled_config['config']['seq_blocks_4']['config'].get('channel_multiplier_input', 1),
            sampled_config['config']['seq_blocks_4']['config'].get('channel_multiplier_output', 1),
            sampled_config['config']['seq_blocks_6']['config'].get('channel_multiplier_input', 1)
        ]

        mg = MaseGraph(self.model) # only pass the architecture but not the weights!
        mg,_ = init_metadata_analysis_pass(mg, None)

        # now revise the network architecture

        if sampled_config is not None:
            #print("\n")
            #print("sampled_config",sampled_config)
            #print("\n")

            # change the network architecture
            mg, _ = self.redefine_linear_transform_pass(mg, sampled_config)
            mg, _ = self.redefine_relu_pass(mg, sampled_config)
            mymodel = load_model(f"/mnt/d/imperial/second_term/adls/rs1923/mase_real/mase_output/4_3/model_with_multiplier_{numbers[0]}_{numbers[1]}_{numbers[2]}_{numbers[3]}.ckpt", "pl", mg.model)
            mg.model = mymodel
            print(mg.model)
        
        self.config = DEFAULT_NETWORK_CONFIG

        return mg

        # only the model architecture at present, later all data from data_loader will be trained on this new architecture
    

    def build_search_space(self):
        """
        Build the search space for the network architecture (only layers that can be modified).
        """
        
        multiplier_options = [1, 2, 3, 4, 5]

        self.choices_flattened = {
            "seq_blocks_2_channel_multiplier_output": multiplier_options, 
            "seq_blocks_4_channel_multiplier_output": multiplier_options,
        }
        
        self.choice_lengths_flattened = {k: len(v) for k, v in self.choices_flattened.items()}


    def flattened_indexes_to_config(self, indexes: dict[str, int]):
        config = {
            "config":{
            "by": "name",
            "default": {"config": {"name": None}},
            "seq_blocks_2": {"config": {"name": "output_only"}},
            "seq_blocks_4": {"config": {"name": "both"}},
            "seq_blocks_6": {"config": {"name": "input_only"}},
            }
        }

        for key, index in indexes.items():
            #pdb.set_trace()
            parts = key.split('_')  # To separate e.g.:'seq_blocks_2', 'channel_multiplier_output'
            block_name = '_'.join(parts[:3]) # e.g., 'seq_blocks_2'
            param_name = '_'.join(parts[3:]) # e.g., 'channel_multiplier_output'

            value = self.choices_flattened[key][index] # value: choice of multiplier

            if "output" in param_name:
                config["config"][block_name]["config"]["channel_multiplier_output"] = value
                # Ensure the input multiplier of the following block matches the output multiplier of current block
                if block_name == "seq_blocks_2":
                    config["config"]["seq_blocks_4"]["config"]["channel_multiplier_input"] = value
                elif block_name == "seq_blocks_4":
                    config["config"]["seq_blocks_6"]["config"]["channel_multiplier_input"] = value

        return config




