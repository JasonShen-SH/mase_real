# Lab1

## Varying Parameters
 We have trained the following 12 models with specific parameters (batch size, maximum number of epochs, learning rate, etc.) for training and validation.
 
|           | max-epoch | learning-rate | batch-size |val_acc_last_epoch|
|-----------|-----------|---------------|------------|------------------|
| model1    | 10        | 1e-5          | 256        |      0.513       |
| model2    | 50        | 1e-5          | 256        |      0.618       |
| model3    | 100       | 1e-5          | 256        |      0.678       |
| model4    | 10        | 1e-5          | 16         |      0.569       |
| model5    | 10        | 1e-5          | 32         |      0.572       |
| model6    | 10        | 1e-5          | 64         |      0.550       |
| model7    | 10        | 1e-5          | 128        |      0.531       |
| model8    | 10        | 1e-5          | 512        |      0.471       |
| model9    | 10        | 1e-6          | 256        |      0.513       |
| model10   | 10        | 1e-4          | 256        |      0.680       |
| model11   | 10        | 1e-3          | 256        |      0.713       |
| model12   | 10        | 1e-2          | 256        |      0.573       |

For each model with specific configuration, we conducted 10 training iterations, each capped at a predefined number of epochs. （To clarify, if each training iteration lasts for 50 epochs, and we conducted 10 separate training iterations, it amounted to a total of 500 epochs across all iterations.)__ 

Finally, we calculated the average validation accuracy using the values obtained at the final epoch of each independent training iteration, and this is represented by "**val_acc_last_epoch**".

### Impact of varying batch sizes
<img src="../../imgs/val_acc_batch_size.png" width=500>
After conducting experiments with batch sizes of 16, 32, 64, 128, 256, and 512, we have derived the following insights:     

Our obeservations indicate that an increase in batch size correlates with a decrease in validation accuracy. This phenomenon can be attributed to the fact that larger batch sizes provide a more comprehensive representation of the training dataset, leading to higher training accuracy. However, this comprehensive representation would normally have much higher training accuracy, which subsequently facilitate convergence towards local minima (as weight updates become less frequent).

Conversely, smaller batch sizes provide gradient estimates with higher variance, which is beneficial in that it enables the model to avoid the local minima, guiding it towards more optimal solutions that are closer to the global minima. In our experiments, generally speaking, smaller batch size resulted in improved validation accuracy due to the more frequent and varied weight updates.

Nonetheless, it is important to note that excessively small batch sizes can introduce an excessive amount of noise into the gradient estimates, which leads to instability during training and therefore, degraded model performance. (In our case, when batch size is only 16, the validation accuracy dropped from 57.2% to 56.9%).

Besides the impact on validation accuracy, we've also found that smaller batch sizes could lead to better generalization ability. See the table below for comparison: 
|           | batch-size |val_acc_last_epoch|train_acc_last_step|
|-----------|------------|------------------|-------------------|
| model4    | 16         |      0.569       |       0.487       |
| model5    | 32         |      0.572       |       0.500       |
| model6    | 64         |      0.550       |       0.500       |
| model7    | 128        |      0.531       |       0.588       |
| model1    | 256        |      0.513       |       0.561       |
| model8    | 512        |      0.471       |       0.487       |   

Here, we use "**train_acc_last_step**" to represent the accuracy of the last training step. (This metric is not particularly accurate due to its randomness, but it could be representative).

It has been observed that larger batch sizes typically yield higher training accuracy compared to validation accuracy. Conversely, this trend tends to invert with the reduction of batch sizes, which means better generalization ability.

### Impact of varying max epochs
<img src="../../imgs/val_acc_max_epoch.png" width=400>

We found that larger total epochs of training result in the model's overall better performance, as more training would capture more patterns of the data. Meanwhile, as the model **JSC-Tiny** used is a simple model with only one linear layer and the dataset is big enough, the overfitting problem did not occur ith more training epochs, as can be observed from the table below:
|           |  max-epoch |val_acc_last_epoch|train_acc_last_step|
|-----------|------------|------------------|-------------------|
| model1    | 10         |      0.513       |       0.561       |
| model2    | 50         |      0.618       |       0.678       |
| model3    | 100        |      0.678       |       0.714       |

In our case, both the validation accuracy and the training accuracy increased with more epochs.

### What is happening with a large learning rate and what is happening with a small learning rate and why? What is the relationship between learning rates and batch sizes?
<img src="../../imgs/val_acc_lr.png" width=500>

With larger learning rate, there's rapid convergence to a solution, and it has the potential to escape the local minima with large step. However, the solution might not be the optimal one (global minima), it mgith instead diverge from the minima point. Moreover, oscillation might occur near optimal points.

With lower learning rate, there's slow convergence due to small steps, and the model might fall into local minima, especially when the batch size is also large.

|           | leanring-rate |val_acc_last_epoch|train_acc_last_step|
|-----------|---------------|------------------|-------------------|
| model9    | 1e-6          |      0.513       |       0.561       |
| model1    | 1e-5          |      0.513       |       0.561       |
| model10   | 1e-4          |      0.680       |       0.643       |
| model11   | 1e-3          |      0.713       |       0.694       |
| model12   | 1e-2          |      0.573       |       0.597       |

In our experiments, the validation accuracy increases from 1e-6 to 1e-3, before falling back to 57.3% at 1e-2. 

**Realtionship between learning rates and batch sizes**:  

Both hyperparameters have impact on the speed of convergence accroding to the formula: 

<img src="../../imgs/weights_update.png" width=180>
Therefore, when batch size is large (indicating smaller gradiant of loss to weights), we usually choose larger learning rate. Vice versa.


## Train your own network

We've implemented a network with the following architecture:

**Legend**:
<div style="display: flex;">
  <figure style="margin-right: 20px;">
    <img src="../../imgs/conv1d.png" width="200"/>
  </figure>
  <figure>
    <img src="../../imgs/residual_block.png" width="200"/>
  </figure>
</div>

<pre>Conv1D Illustration               Residual Block Diagram   </pre>


**Network Architecture**

<img src="../../imgs/jsc_rs1923.png" width="300" />

The network begins with an input of size 16 and features a residual block after two initial 1D convolutional and ReLU layers, the parallel paths of residual block then converge to a linear layer, and finally produces a final output of size 5 after a sigmoid activation function.

**Parameters**
<pre>
jsc_rs1923 = JSC_rs1923(info)
total_params = 0
for param in jsc_rs1923.parameters():
    total_params += param.numel()
print(f'Total number of JSC_rs1923 parameters: {total_params}')
 # same for jsc_tiny
</pre>
And we get the parameters of jsc_rs1923 to be 3285, compared to that of JSC-Tiny (117).

**Performance**

Note that in order to train the modified network, we first need to modify the dataset as follows:
<pre>
# ./chop/dataset/physical/jsc.py  Line 164
def __getitem__(self,idx):
   x = self.X[idx]
   y = self.Y[idx]
   x = x.unsqueeze(0)  # used only for the convolution model JSC_rs1923
   return x,y
# This adds an extra dimension to the input data, changing it from 16 to 1x16.
</pre>

We've trained for 10 epochs and derived the validation accuracy of the last epoch at **65.1%**, which is higher than that of JSC-tiny at **51.3%**.
<pre>
Epoch 9: 100% 3084/3084 [00:33<00:00, 91.36it/s, v_num=2, train_acc_step=0.643, val_acc_epoch=0.651, val_loss_epoch=1.020]
INFO     Training is completed
</pre>

To summarize, the integration of 1D convolutional layers and residual modules has facilitated an augmented model architecture. This augmented model has the capacity for more profound feature extraction, which contributes to a better overall performance.




# Lab2

## 1. Explain the functionality of report_graph_analysis_pass and its printed jargons such as placeholder, get_attr ... 

The **report_graph_analysis_pass** function provides a high-level abstraction of the model's architecture. By utilizing the torch.fx module's capabilities[1], it categorizes the model's various components into distinct types, and enumerate each component along with its respective parameters. 

Furthermore, though no transform performed at the moment, this function offers strategic insights into potential model transformation methodologies for optimization.

[1]: The function mainly iterates nodes in graph.fx_graph, and graph.fx_graph is defined in:
<pre>
# ./chop/ir/graph/mase_graph.py
self.model = fx.GraphModule(model, self.tracer.trace(model, cf_args))  # fx is torch.fx
''''''
def fx_graph(self):
       return self.model.graph
# Therefore, **report_graph_analysis_pass** utilizes torch.fx's graph to depict the model.
</pre>

I interpret the following printed jargons as different types of nodes, where each node is a data structure that represents individual operations within a torch.fx Graph[2].

**placeholder**: The input of the model(function), the variable that gives target's value is denoted as **x**(target=x).

**get_attr**: Retrieves parameters from modules, in our case, if we explicitly retrives for weights and bias of linear layer:
<pre>
linear_weight = self.seq_blocks[2].weight (self.seq_blocks[2] is the linear module node)
linear_bias = self.seq_blocks[2].bias
</pre>
Then there will be nodes with node_op: get_attr.

**call_function**: Mainly free functions in Python.

**call_method**: This type of node represents the invocation of a method on a particular type of object. Take **tensor** in deep learning as instance, the commonly-seen operations such as tensor.view(), tensor.reshape(), tensor.tonumpy() are all "call_method".

<pre>
If there are two tensors a and b, then:
a+b is "call_function",
a.add(b) is "call_method" as it uses the add() method of tensor.
</pre>

**call_module**: These modules are instances of **torch.nn.Module**, such as nn.Linear, nn.Conv1d, nn.ReLU, nn.Batchnorm and so on.
Note that our self-defined class is also call_module as long as it's sub-class of torch.nn.Module, such as :
<pre>
Class Self_Defined(nn.module):
{
'''
Self-defined structure
'''
}
</pre>


## 2. What are the functionalities of profile_statistics_analysis_pass and report_node_meta_param_analysis_pass respectively?

The pass **profile_statistics_analysis_pass** is dedicated to the systematic extraction of quantitative metrics pertaining to the specified modules within the computational graph (e.g., parameters & activations). These specific modules are typically delineated through a configuration schema.

In our context, the emphasis is on the weight statistics (namely, the mean and variance) for the Linear module, alongside the activation metrics (including maximum, minimum, and the overall range) for the ReLU module.

The datails of the computing and updating process of these metrics can be found within **stat.py** located within the same directory. 

The function **report_node_meta_param_analysis_pass respectively** traverses each node in the graph and reports their metadata, these metadata could be **node name, node op, mase type, mase op, and common & hardware & software parameters** that we've defined.

In our specific instance, we spolight **software metadata** in conjunction with fundamental metadata.

Take seq_blocks_2 (linear) as an instance:
<pre>
 Node name    | Fx Node op   | Mase type           | Mase op      | Software Param                                                                           |
+==============+==============+=====================+==============+==========================================================================================+
| seq_blocks_2 | call_module  | module_related_func | linear       | {'args': {'bias': {'stat': {'variance_precise': {'count': 5,                             |
|              |              |                     |              |                                                  'mean': -0.010053220205008984,          |
|              |              |                     |              |                                                  'variance': 0.05255531892180443}}},     |
|              |              |                     |              |           'data_in_0': {'stat': {}},                                                     |
|              |              |                     |              |           'weight': {'stat': {'variance_precise': {'count': 80,                          |
|              |              |                     |              |                                                    'mean': -0.009381229057908058,        |
|              |              |                     |              |                                                    'variance': 0.020840153098106384}}}}, |
|              |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                    
</pre>

We could also showcase **common metadata** in conjunction with fundamental metadata:
<pre>
 +--------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------
| Node name    | Fx Node op   | Mase type           | Mase op      | Common Param                                                                                                   
+==============+==============+=====================+==============+===========================================================================================
| seq_blocks_2 | call_module  | module_related_func | linear       | { 'args': 
                                                                       {
                                                                         'bias': {
                                                                           'from': None,
                                                                           'precision': [32],
                                                                           'shape': [5],
                                                                           'type': 'float',
                                                                           'value': '......'
                                                                         },
                                                                         'data_in_0': {
                                                                           'precision': [32],
                                                                           'shape': [8,16],
                                                                           'torch_dtype': 'torch.float32',
                                                                           'type': 'float',
                                                                           'value': '......'
                                                                         },
                                                                         'weight': {
                                                                           'from': None,
                                                                           'precision': [32],
                                                                           'shape': [16,5],
                                                                           'type': 'float',
                                                                           'value': '......'
                                                                         }
                                                                       },
                                                                       'mase_op': 'linear',
                                                                       'mase_type': 'module_related_func',
                                                                       'results': {
                                                                         'data_out_0': {
                                                                           'precision': [32],
                                                                           'shape': [8,5],
                                                                           'torch_dtype': 'torch.float32',
                                                                           'type': 'float',
                                                                           'value': '......'
                                                                         }
                                                                       }
                                                                     }
</pre>
In this case, more information including 'bias', 'input data', 'weight', and 'output data' can be derived.


## 3. Explain why only 1 OP is changed after the quantize_transform_pass

As we only include the **node representing linear module** to be transformed in the configuration file below, only that node is transformed.
<pre>
pass_args = {
"by": "type",
"default": {"config": {"name": None}},
"linear": {  
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
},
}
</pre>
No other node has been transformed except the node representing linear module.  

Therefore, only 1OP (operation) is changed.


## 4. Write some code to traverse both mg and ori_mg, check and comment on the nodes in these two graphs. 

The code to traverse nodes and check their attributes:
<pre>
def get_type_str(node):
    if node.op == "call_module": % batch_norm, relu, linear, relu
        return type(get_node_actual_target(node)).__name__
    else: % placeholder, output
        return node.target  
 
logger = logging.getLogger(__name__)
headers = [ "Ori name", "New name", "Node_OP", "MASE_TYPE", "Mase_OP", "Original type", "Quantized type", "Changed"]
rows=[]
for ori_n, n in zip(ori_mg.fx_graph.nodes, mg.fx_graph.nodes):
    rows.append(
            [
                ori_n.name,
                n.name,
                n.op,
                get_mase_type(n),
                get_mase_op(n),
                get_type_str(ori_n),
                get_type_str(n),
                type(get_node_actual_target(n)) != type(get_node_actual_target(ori_n)),
            ]
        )
df = pd.DataFrame(rows, columns=headers)
</pre>

<img src="../../imgs/2_4.png" width="800" />

We could find from the dataframe that only the node representing linear module has changed (i.e.quantized), the original type is *Linear*, the quantized type is *LinearInteger*.


## 5. Perform the same quantisation flow to the bigger JSC network that you have trained in lab1.

We've performed the same quantisation flow to our self-designed JSC network in lab1. As it contains convolution layer, we customize <code>pass-args</code> as follows:
<pre>
pass_args = {
"by": "type",
"default": {"config": {"name": None}},

"conv1d": {
    "config": {
        "name": "integer",
        # data
        "data_in_width": 8,
        "data_in_frac_width": 4,
        # weight
        "weight_width": 8,
        "weight_frac_width": 4,
        # bias
        "bias_width": 8,
        "bias_frac_width": 4,
    }
},

"linear": {  
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
},
}
</pre>

In <code>pass-args</code>, for the convolution layer, we set its quantized <code>weight_width</code> to be 8 and <code>weight_frac_width</code> to be 4. The same quantization apply to its data and bias. 

The quantized table:

<img src="../../imgs/2_5_quantized_table.png" width="800"/>

Only the convolution and linear layers have been quantized. Specifically, for convolution layers, their quantized type is Conv1dInteger, while for the linear layer, its quantized type is LinearInteger.

The quantized histogram provides a more direct observation by looking at different node types:

<img src="../../imgs/2_5_quantized_histogram.png" width="600"/>



## 6. Write code to show and verify that the weights of these layers are indeed quantised. 

**Before showing the code for quantisation verification, we first illustrate its quantisation method:**

There are two main methods of neural network quantisation: Post-Training Quantisation (PTQ) and Quantization Aware Training (QAT).

We've found that the quantisation in our task is Post-Training Quantisation (PTQ) from the following code:

<pre>
# The quantizer for w,x,b is defined in *./quantize/quantized_modules/linear.py*
self.w_quantizer = partial(integer_quantizer, width=w_width, frac_width=w_frac_width)
self.x_quantizer = partial(integer_quantizer, width=x_width, frac_width=x_frac_width)
self.b_quantizer = partial(integer_quantizer, width=b_width, frac_width=b_frac_width)

# where integer_quantizer is defined in *./quantize/quantize/quantizers/integer.py*
if frac_width is None:
    frac_width = width // 2
if is_signed:
    int_min = -(2 ** (width - 1))
    int_max = 2 ** (width - 1) - 1
else:
    int_min = 0
    int_max = 2**width - 1
scale = 2**frac_width

if isinstance(x, (Tensor, ndarray)):
    return my_clamp(my_round(x.mul(scale)), int_min, int_max).div(scale)
elif isinstance(x, int):
    return x
else:
    return my_clamp(my_round(x * scale), int_min, int_max) / scale
</pre>

<pre>
For (w_width,w_frac_width) = (8,4), w_quantized = (round(w*16))/16.

Same quantization process apply to data x and bias b, which is from FP32 to INT8.

No training process is performed during training. 
</pre>

Therefore, we could conclude that our quantization process is **Post-Training Quantisation (PTQ)**.

Basically, PTQ contains three main steps: 

1) **Full-Precision Training Phase**: The neural network is trained utilizing full-precision weights (typically in FP32 format) to ensure the learning algorithm can capture the full range of data features and complexities without precision loss.

2) **Quantization Procedure**: Post training, a quantization algorithm is applied to the network parameters: weights (w), biases (b), and activations (x). This process involves converting these parameters from full-precision (FP32) to a lower precision format (commonly INT8). (Techniques such as calibration might be used to minimize information loss)

3) **Quantized Inference Evaluation**: With the quantized model, the inference process is executed using the validation dataset to assess the impact of quantization on model accuracy.

To summarize, thw whole idea is **a trade-off between computational efficiency and predictive performance**.

The PTQ process is summarized in the two figures below:

<img src="../../imgs/ptq_training_quantisation.png" width="700" />

<img src="../../imgs/ptq_inference.png" width="600" />

**Now we write code for quantisation verification**

<pre>
for ori_n, n in zip(ori_mg.fx_graph.nodes, mg.fx_graph.nodes):
    # As we've seen, the convolution and linear modules have changed
    if isinstance(get_node_actual_target(ori_n), torch.nn.modules.Linear): # Linear
        print(f"There is quantization at {n.name}, mase_op: {get_mase_op(n)}")
        print(f"original module: {type(get_node_actual_target(ori_n))}, new_module: {type(get_node_actual_target(n))}")
        print(f"original weight: {get_node_actual_target(ori_n).weight}")
        print(f"quantized weight: {get_node_actual_target(n).w_quantizer(get_node_actual_target(n).weight)}")
        print(f"original bias: {get_node_actual_target(ori_n).bias}")
        print(f"quantized bias: {get_node_actual_target(n).b_quantizer(get_node_actual_target(n).bias)}")

        # generate a random input to a quantized layer for quantisation verification
        random_input = torch.randn(get_node_actual_target(n).in_features)
        print(f'output for original module: {get_node_actual_target(ori_n)(random_input)}')
        print(f'output for quantized module: {get_node_actual_target(n)(random_input)}')
   
    if isinstance(get_node_actual_target(ori_n), torch.nn.modules.conv.Conv1d): # Conv1d
        print(f"There is quantization at {n.name}, mase_op: {get_mase_op(n)}")
        print(f"original module: {type(get_node_actual_target(ori_n))}, new_module: {type(get_node_actual_target(n))}")
        print(f"original weight: {get_node_actual_target(ori_n).weight}")
        print(f"quantized weight: {get_node_actual_target(n).w_quantizer(get_node_actual_target(n).weight)}")
        print(f"original bias: {get_node_actual_target(ori_n).bias}")
        print(f"quantized bias: {get_node_actual_target(n).b_quantizer(get_node_actual_target(n).bias)}")
</pre>

Given the extensive number of parameters associated with conv1d layers, we have selected the linear layer as a representative example to illustrate the outcomes of the quantization process.
<pre>
There is quantization at fc, mase_op: linear
original module: <class 'torch.nn.modules.linear.Linear'>, new_module: <class 'chop.passes.graph.transforms.quantize.quantized_modules.linear.LinearInteger'>
original weight: Parameter containing:
tensor([[ 0.0752, -0.0539,  0.0252,  ...,  0.0037, -0.1073,  0.0500],
        [-0.0059,  0.0253,  0.0007,  ...,  0.0487,  0.0635,  0.0750],
        [ 0.0975,  0.0499, -0.0471,  ...,  0.0235,  0.0838,  0.1013],
        [ 0.0061,  0.0469, -0.0265,  ...,  0.0446,  0.0610, -0.0798],
        [ 0.0059, -0.0739,  0.0598,  ...,  0.0549, -0.0014, -0.0054]],
       requires_grad=True)
quantized weight: tensor([[ 0.0625, -0.0625,  0.0000,  ...,  0.0000, -0.1250,  0.0625],
        [-0.0000,  0.0000,  0.0000,  ...,  0.0625,  0.0625,  0.0625],
        [ 0.1250,  0.0625, -0.0625,  ...,  0.0000,  0.0625,  0.1250],
        [ 0.0000,  0.0625, -0.0000,  ...,  0.0625,  0.0625, -0.0625],
        [ 0.0000, -0.0625,  0.0625,  ...,  0.0625, -0.0000, -0.0000]],
       grad_fn=<IntegerQuantizeBackward>)
original bias: Parameter containing:
tensor([-0.0223,  0.0642,  0.0230,  0.0888, -0.0599], requires_grad=True)
quantized bias: tensor([-0.0000,  0.0625,  0.0000,  0.0625, -0.0625],
grad_fn=<IntegerQuantizeBackward>)
</pre>

We can observe that while the original weights are arbitrary floating-point numbers with no discernible distribution, the quantized weights are clearly quantized floating-point numbers, such as ±0.125, ±0.625, 0, and so forth. In other words, the quantized weights can be represented by a set containing a finite number of floating-point numbers, thereby reducing the precision required to represent this set. So did bias.

We also randomly generated an input for the linear layer and passed it through both the original linear layer and the weight-quantized linear layer. It can be observed in the figure below that their outputs are different.
<pre>
output for original module: tensor([ 0.3859, -0.7512, -0.4924, -0.3702,  0.4731], grad_fn=<ViewBackward0>)
output for quantized module: tensor([ 0.2188, -0.2188, -0.8008, -0.3789,  0.3516], grad_fn=<ViewBackward0>)
</pre>

This principle and observation are equally applicable to conv1d layers.


## 7.Load your own pre-trained JSC network, and perform perform the quantisation using the command line interface.

We developed <code>jsc_rs1923_by_type.toml</code> for our pre-trained JSC-rs1923 network, extending <code>jsc_toy_by_type.toml</code>. This includes essential modifications for network adaptation like the following:
<pre>
[passes.quantize.conv1d.config]
name = "integer"
"data_in_width" = 8
"data_in_frac_width" = 4
"weight_width" = 8
"weight_frac_width" = 4
"bias_width" = 8
"bias_frac_width" = 4
</pre>

Run command:
<pre>
!./ch transform --config configs/examples/jsc_rs1923_by_type.toml --task cls --cpu=0
</pre>

We could the same quantisation result:

<img src="../../imgs/2_7.png" width="600" />


## Optional Task: Write your own pass

We calculate the FLOPs and BitOPs within a **single batch** for **JSC-Tiny**.

To demonstrate, we selected JSC-Tiny as our model, comprising three distinct layer types: Linear, BatchNorm, and ReLU.

Within the file <code>chop.passes.graph.analysis.add_metadata.count_flop_bitops.py</code>, we introduced a new pass <code>add_flops_bitops_analysis_pass</code>.

Within the pass, we compute the FLOPs for Linear, BatchNorm, and ReLU layers, employing specific computational strategies delineated in the subsequent functions:
(The specific methods for calculating FLOPs are detailed in the comments within the functions)

For batchnorm, note that γ and β here are 1*16 vector, and the multiplication and bias terms involve element-wise multiplication.
<img src="../../imgs/batch_norm.png" width="200" />

<pre>
def calculate_flop_for_linear(module, batch_size):
    in_features = module.in_features
    out_features = module.out_features
    return batch_size*(in_features * out_features)
 
def calculate_flop_for_batchnorm1d(module, batch_size):
    num_features = module.num_features
    # calculate the mean: num_features * batch_size  [for each element, (batch_size-1)add, 1division]
    # calculate the variance: (2*num_features+(num_features-1))*batch_size + (batch_size-1)  [for each element:2, for each sample: 2*num_features+(num_features-1)]
    # calculate the denominator (knowing variance): 2  [add bias & square root]
    # calculate for each sample xi: 4*num_features  [for each element, 4: 1*minus, 1*division, 1*multiply, 1*add]
    return num_features * batch_size + (2*num_features+(num_features-1))*batch_size + (batch_size-1) + 2 + batch_size*(4*num_features)

def calculate_flop_for_relu(module, input_features, batch_size):
    # per element comparison with 0 (in essence, a minus)
    input_features = input_features*batch_size # relu's input_feature need to be passed in
    return input_features
</pre>

Within the pass, we also compute the BitOPs for Linear, employing specific computational strategy as below:
<pre>
def calculate_bitwise_ops_for_linear(module, data_bit_width, weight_bit_width, bias_bit_width, batch_size):
    in_features = module.in_features
    out_features = module.out_features
    bitwise_ops_per_multiplication = data_bit_width * weight_bit_width
    bitwise_ops_per_addition = data_bit_width * weight_bit_width
    bitwise_ops_per_output_feature = in_features * bitwise_ops_per_multiplication + (in_features - 1) * bitwise_ops_per_addition
    if module.bias is not None:
        bitwise_ops_per_output_feature += bias_bit_width
    total_bitwise_ops = out_features * bitwise_ops_per_output_feature
    return total_bitwise_ops*batch_size
</pre>

Finally, we run the command back to the main <code>.ipynb</code> under <code>mase</code>:
<pre>
from chop.passes.graph.analysis import add_flops_bitops_analysis_pass
mg = MaseGraph(model=own_model)
mg, _ = init_metadata_analysis_pass(mg, None)
mg, total_flops, total_bitops, _ = add_flops_bitops_analysis_pass(mg)
</pre>

And we get:  (Note again: We calculate the FLOPs and BitOPs within a **single batch** for **JSC-Tiny**).
<pre>
# batch_size = 8
total_flops = 1833
total_bitops = 79680

# batch_size = 512
total_flops = 117249
total_bitops = 5099520
</pre>





