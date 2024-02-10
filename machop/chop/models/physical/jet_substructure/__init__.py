"""
Jet Substructure Models used in the LogicNets paper
"""

import torch.nn as nn
import torch.nn.functional as F

from chop.models.utils import MaseModelInfo


class JSC_Toy(nn.Module):
    def __init__(self, info):
        super(JSC_Toy, self).__init__()
        self.seq_blocks = nn.Sequential(
            # 1st LogicNets Layer
            nn.BatchNorm1d(16),  # input_quant       # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 8),  # linear              # 2
            nn.BatchNorm1d(8),  # output_quant       # 3
            nn.ReLU(8),  # 4
            # 2nd LogicNets Layer
            nn.Linear(8, 8),  # 5
            nn.BatchNorm1d(8),  # 6
            nn.ReLU(8),  # 7
            # 3rd LogicNets Layer
            nn.Linear(8, 5),  # 8
            nn.BatchNorm1d(5),  # 9
            nn.ReLU(5),
        )

    def forward(self, x):
        return self.seq_blocks(x)


class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self, info):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # unchanged      
            nn.ReLU(16),     
            nn.Linear(16, 16), # output      
            nn.ReLU(16),     
            nn.Linear(16, 16), # input & output     
            nn.ReLU(16),        
            nn.Linear(16, 5),  # input    
            nn.ReLU(5),       
        )

    def forward(self, x):
        return self.seq_blocks(x)
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual  
        out = F.relu(out)
        return out

class JSC_rs1923(nn.Module):
    def __init__(self, info):
        super(JSC_rs1923, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1)
        self.block = ResBlock(16, 16)
        self.fc = nn.Linear(16 * 16, 5)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.block(x)
        x = x.view(-1, 16 * 16) 
        x = self.fc(x)
        x = F.relu(x)
        return x
    

class JSC_Tiny(nn.Module):
    def __init__(self, info):
        super(JSC_Tiny, self).__init__()
        self.seq_blocks = nn.Sequential(
            # 1st LogicNets Layer
            nn.BatchNorm1d(16),  # input_quant       # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 5),  # linear              # 2
            # nn.BatchNorm1d(5),  # output_quant       # 3
            nn.ReLU(5),  # 4
        )

    def forward(self, x):
        return self.seq_blocks(x)


class JSC_S(nn.Module):
    def __init__(self, info):
        super(JSC_S, self).__init__()
        self.config = info
        self.num_features = self.config["num_features"]
        self.num_classes = self.config["num_classes"]
        hidden_layers = [64, 32, 32, 32]
        self.num_neurons = [self.num_features] + hidden_layers + [self.num_classes]
        layer_list = []
        for i in range(1, len(self.num_neurons)):
            in_features = self.num_neurons[i - 1]
            out_features = self.num_neurons[i]
            bn = nn.BatchNorm1d(out_features)
            layer = []
            if i == 1:
                bn_in = nn.BatchNorm1d(in_features)
                in_act = nn.ReLU()
                fc = nn.Linear(in_features, out_features)
                out_act = nn.ReLU()
                layer = [bn_in, in_act, fc, bn, out_act]
            elif i == len(self.num_neurons) - 1:
                fc = nn.Linear(in_features, out_features)
                out_act = nn.ReLU()
                layer = [fc, bn, out_act]
            else:
                fc = nn.Linear(in_features, out_features)
                out_act = nn.ReLU()
                layer = [fc, out_act]
            layer_list = layer_list + layer
        self.module_list = nn.ModuleList(layer_list)

    def forward(self, x):
        for l in self.module_list:
            x = l(x)
        return x


# Getters ------------------------------------------------------------------------------
def get_jsc_toy(info):
    # TODO: Tanh is not supported by mase yet
    return JSC_Toy(info)


def get_jsc_tiny(info):
    return JSC_Tiny(info)


def get_jsc_s(info):
    return JSC_S(info)


def get_jsc_three_linear_layers(info):
    return JSC_Three_Linear_Layers(info)


def get_jsc_rs1923(info):
    return JSC_rs1923(info)


info =  MaseModelInfo(
            "jsc-rs1923",
            model_source="physical",
            task_type="physical",
            physical_data_point_classification=True,
            is_fx_traceable=True,
        )
jsc_rs1923 = JSC_rs1923(info)
total_params = 0
for param in jsc_rs1923.parameters():
    total_params += param.numel()
print(f'Total number of JSC_1923 parameters: {total_params}')


info =  MaseModelInfo(
            "jsc-tiny",
            model_source="physical",
            task_type="physical",
            physical_data_point_classification=True,
            is_fx_traceable=True,
        )
jsc_tiny = JSC_Tiny(info)
total_params = 0
for param in jsc_tiny.parameters():
    total_params += param.numel()
print(f'Total number of JSC_Tiny parameters: {total_params}')