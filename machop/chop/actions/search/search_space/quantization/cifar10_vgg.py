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

DEFAULT_NETWORK_CONFIG = { 
    "config": {
        "by":"name",
        "default": {"config": {"name": None}},
        "seq_blocks_0": {
            "config": {
                "name": "output_only",
                "channel_output": 128,
                }
            },
        "seq_blocks_1": {
            "config": {
                "name": "bn",
                "channel": 128,
                }
            },
        "seq_blocks_2": {
            "config": {
                "name": "relu",
                }
            },

        "seq_blocks_3": {
            "config": {
                "name": "both",
                "channel_input": 128,
                "channel_output": 128,
                }
            },
        "seq_blocks_4": {
            "config": {
                "name": "bn",
                "channel": 128,
                }
            },
        "seq_blocks_5": {
            "config": {
                "name": "relu",
                }
            },
        "seq_blocks_6": {
            "config": {
                "name": "maxpool",
                }
            },

        "seq_blocks_7": {
            "config": {
                "name": "both",
                "channel_input": 128,
                "channel_output": 256,
                }
            },
        "seq_blocks_8": {
            "config": {
                "name": "bn",
                "channel": 256,
                }
            },
        "seq_blocks_9": {
            "config": {
                "name": "relu",
                }
            },

        "seq_blocks_10": {
            "config": {
                "name": "both",
                "channel_input": 256,
                "channel_output": 256,
                }
            },
        "seq_blocks_11": {
            "config": {
                "name": "bn",
                "channel": 256,
                }
            },
        "seq_blocks_12": {
            "config": {
                "name": "relu",
                }
            },
        "seq_blocks_13": {
            "config": {
                "name": "maxpool",
                }
            },

        "seq_blocks_14": {
            "config": {
                "name": "both",
                "channel_input": 256,
                "channel_output": 512,
                }
            },
        "seq_blocks_15": {
            "config": {
                "name": "bn",
                "channel": 512,
                }
            },
        "seq_blocks_16": {
            "config": {
                "name": "relu",
                }
            },

        "seq_blocks_17": {
            "config": {
                "name": "both",
                "channel_input": 512,
                "channel_output": 512,
                }
            },
        "seq_blocks_18": {
            "config": {
                "name": "bn",
                "channel": 512,
                }
            },
        "seq_blocks_19": {
            "config": {
                "name": "relu",
                }
            },
        "seq_blocks_20": {
            "config": {
                "name": "maxpool",
                }
            },

        "seq_blocks_22": {
            "config": {
                "name": "linear",
                "channel_input": 8192,
            }
        },
    }   
}


class VggSearch(SearchSpaceBase):

    def _post_init_setup(self):
        self.model.to("cpu") 
        self.mg = None 
        self._node_info = None 
        self.default_config = DEFAULT_NETWORK_CONFIG

        # change network architecture by layer name or layer type
        assert (
            "by" in self.config["setup"]
        ), "Must specify entry `by` (config['setup']['by] = 'name' or 'type')"

    
    def instantiate_conv2d(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def redefine_conv2d_transform_pass(self, graph, pass_args=None):
        pass_args_copy = deepcopy(pass_args)
        main_config = pass_args_copy.pop('config')
        default = main_config.pop('default', None)
        if default is None:
            raise ValueError("default value must be provided.")
        for node in graph.fx_graph.nodes:
            config = main_config.get(node.name, default)['config']
            name = config.get("name", None)
            if name == "output_only" or name == "both":
                ori_module = graph.modules[node.target]
                if isinstance(ori_module, nn.Conv2d): # Ensure the node is a Conv2d layer 
                    in_channels = ori_module.in_channels
                    out_channels = ori_module.out_channels
                    if name == "output_only":
                        out_channels = config["channel_output"]
                    elif name == "both":
                        in_channels = config["channel_input"]
                        out_channels = config["channel_output"]      
                    new_module = self.instantiate_conv2d(
                        in_channels, out_channels
                    )
                    parent_name, name = get_parent_name(node.target)
                    setattr(graph.modules[parent_name], name, new_module)
        return graph, {}
    

    def instantiate_bn(self, num_features):
        return nn.BatchNorm2d(
            num_features=num_features,
        )

    def redefine_bn_transform_pass(self, graph, pass_args=None):
        pass_args_copy = deepcopy(pass_args)
        main_config = pass_args_copy.pop('config')
        default = main_config.pop('default', None)
        if default is None:
            raise ValueError("default value must be provided.")
        for node in graph.fx_graph.nodes:
            config = main_config.get(node.name, default)['config']
            name = config.get("name", None)
            if name == "bn":
                ori_module = graph.modules[node.target]
                if isinstance(ori_module, nn.BatchNorm2d): # Ensure the node is a BatchNorm2d layer 
                    #num_features = ori_module.num_features
                    if 'channel' in config:
                        num_features = config['channel']
                    new_module = self.instantiate_bn(
                        num_features
                    )
                    parent_name, name = get_parent_name(node.target)
                    setattr(graph.modules[parent_name], name, new_module)
        return graph, {}
    

    def instantiate_linear(self, in_features, out_features, bias):
        if bias is not None:
            bias = True
        return nn.Linear(
            in_features=in_features, 
            out_features=1024,
            bias=bias
        )

    def redefine_linear_transform_pass(self, graph, pass_args=None):
        # graph = self.mg
        pass_args_copy = deepcopy(pass_args)
        main_config = pass_args_copy.pop('config')
        default = main_config.pop('default', None)
        if default is None:
            raise ValueError(f"default value must be provided.")
        for node in graph.fx_graph.nodes: 
            config = main_config.get(node.name, default)['config']
            name = config.get("name", None)
            if name == "linear":
                ori_module = graph.modules[node.target]
                if isinstance(ori_module, nn.Linear): # Ensure the node is a linear layer 
                    in_features = ori_module.in_features
                    if "channel_input" in config:
                        in_features = config['channel_input'] # this is "="
                        new_module = self.instantiate_linear(in_features, 1024, True)
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
        for node in graph.fx_graph.nodes:
            config = main_config.get(node.name, default)['config']
            name = config.get("name", None)
            if name == "relu":
                new_module = self.instantiate_relu(True)
                parent_name, name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)
        return graph, {}
    

    def instantiate_maxpool(self):
        return nn.MaxPool2d((2,2))

    def redefine_pooling_transform_pass(self, graph, pass_args=None):
        pass_args_copy = deepcopy(pass_args)
        main_config = pass_args_copy.pop('config')
        default = main_config.pop('default', None)
        if default is None:
            raise ValueError("default value must be provided.")
        
        for node in graph.fx_graph.nodes:
            config = main_config.get(node.name, default)['config']
            name = config.get("name", None)
            
            if name == "maxpool":
                ori_module = graph.modules[node.target]
                if isinstance(ori_module, nn.MaxPool2d):
                    new_module = self.instantiate_maxpool()
                    parent_name, name = get_parent_name(node.target)
                    setattr(graph.modules[parent_name], name, new_module)
                    
        return graph, {}




    def rebuild_model(self, sampled_config, is_eval_mode: bool = True):
        if is_eval_mode:
            self.model.eval()
        else:
            self.model.train()
            
        mg = MaseGraph(self.model)
        mg,_ = init_metadata_analysis_pass(mg, None)

        # now revise the network architecture
        if sampled_config is not None:
            #print("\n")
            #print("sampled_config",sampled_config)
            #print("\n")

            # change the network architecture
            mg, _ = self.redefine_conv2d_transform_pass(mg, sampled_config)
            mg, _ = self.redefine_bn_transform_pass(mg, sampled_config)
            mg, _ = self.redefine_relu_pass(mg, sampled_config)
            mg, _ = self.redefine_pooling_transform_pass(mg, sampled_config)
            mg, _ = self.redefine_linear_transform_pass(mg, sampled_config)
            print("new model",mg.model)
        
        # self.config = DEFAULT_NETWORK_CONFIG

        return mg
    

    def build_search_space(self):
        """
        Build the search space for the network architecture (only layers that can be modified).
        """
        
        options = [64, 128, 256, 512]

        self.choices_flattened = {
            "seq_blocks_0_channel_output": options, 
            "seq_blocks_3_channel_output": options,
            "seq_blocks_7_channel_output": options,
            "seq_blocks_10_channel_output": options,
            "seq_blocks_14_channel_output": options,
            "seq_blocks_17_channel_output": options,
        }
        
        self.choice_lengths_flattened = {k: len(v) for k, v in self.choices_flattened.items()}
    
    def flattened_indexes_to_config(self, indexes: dict[str, int]):
        config = {
            "config": {
                "by": "name",
                "default": {"config": {"name": None}},
                "seq_blocks_0": {"config": {"name": "output_only"}},  # 0
                "seq_blocks_1": {"config": {"name": "bn"}},
                "seq_blocks_2": {"config": {"name": "relu"}},

                "seq_blocks_3": {"config": {"name": "both"}},  # 3
                "seq_blocks_4": {"config": {"name": "bn"}},
                "seq_blocks_5": {"config": {"name": "relu"}},
                "seq_blocks_6": {"config": {"name": "maxpool"}},

                "seq_blocks_7": {"config": {"name": "both"}},   # 7
                "seq_blocks_8": {"config": {"name": "bn"}},
                "seq_blocks_9": {"config": {"name": "relu"}},

                "seq_blocks_10": {"config": {"name": "both"}},   # 10
                "seq_blocks_11": {"config": {"name": "bn"}},
                "seq_blocks_12": {"config": {"name": "relu"}},
                "seq_blocks_13": {"config": {"name": "maxpool"}},

                "seq_blocks_14": {"config": {"name": "both"}},   # 14
                "seq_blocks_15": {"config": {"name": "bn"}},
                "seq_blocks_16": {"config": {"name": "relu"}},

                "seq_blocks_17": {"config": {"name": "both"}},   # 17
                "seq_blocks_18": {"config": {"name": "bn"}},
                "seq_blocks_19": {"config": {"name": "relu"}},
                "seq_blocks_20": {"config": {"name": "maxpool"}},
                
                "seq_blocks_22": {"config": {"name": "linear"}},
            }
        }

        for key, index in indexes.items():
            parts = key.split('_') # split layer_name and its parameters
            layer_name = '_'.join(parts[:3])  # e.g., 'seq_blocks_1'
            param_name = '_'.join(parts[3:])  # e.g., 'channel_multiplier_output'

            value = self.choices_flattened[key][index]  # value: choice of multiplier
            config["config"][layer_name]["config"][param_name] = value

        layer_sequence = ["seq_blocks_0", "seq_blocks_3", "seq_blocks_7",
                        "seq_blocks_10", "seq_blocks_14", "seq_blocks_17"]

        bn_indexes = {
            "seq_blocks_0": "seq_blocks_1",  
            "seq_blocks_3": "seq_blocks_4", 
            "seq_blocks_7": "seq_blocks_8",  
            "seq_blocks_10": "seq_blocks_11",
            "seq_blocks_14": "seq_blocks_15",
            "seq_blocks_17": "seq_blocks_18",
        }

        # notice1: Ensure the number of channels kept at least non-decreasing
        for i in range(len(layer_sequence) - 1):
            current_layer = layer_sequence[i]
            next_layer = layer_sequence[i + 1]
            if config["config"][next_layer]["config"]["channel_output"] < config["config"][current_layer]["config"]["channel_output"]:
                config["config"][next_layer]["config"]["channel_output"] = config["config"][current_layer]["config"]["channel_output"]

        # notice2: Batchnorm
        for i in range(len(layer_sequence)):
            current_layer = layer_sequence[i]
            bn_index = int((layer_sequence[i].split("_",2))[2])+1
            print("bn_index",bn_index)
            config["config"][f"seq_blocks_{bn_index}"]["config"]["channel"] = config["config"][current_layer]["config"]["channel_output"]

        # notice3: Ensure the input multiplier of the following block matches the output multiplier of current block
        for i in range(len(layer_sequence) - 1):
            current_layer = layer_sequence[i]
            next_layer = layer_sequence[i + 1]
            config["config"][next_layer]["config"]["channel_input"] = config["config"][current_layer]["config"]["channel_output"]

        # notice4: Ensure the first linear layer after flatten matches the feature dimension
        config["config"]["seq_blocks_22"]["config"]["channel_input"] = (config["config"]["seq_blocks_17"]["config"]["channel_output"]) * (4 * 4)    # 4*4 is the feature map size

        return config





