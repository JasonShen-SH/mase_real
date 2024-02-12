# Lab 3

## 1. Explore additional metrics that can serve as quality metrics for the search process. 

Overall, we have explored four additional metrics: 1)Latency, 2)Model size, 3)FLOPs, and 4)Bit-wise operations.


**Latency**: (Unit: ms, we have multiplied by 1000)

For each search option, we calculate the latency for each input-batch, then accumulate all latencies to take the average.
<pre>
for i, config in enumerate(search_spaces):
    mg, _ = quantize_transform_pass(mg, config)
    ''''''
    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        start_time = time.time() # start time
        preds = mg.model(xs)
        end_time = time.time() # end time
        latency = end_time - start_time # latency
        latencies.append(latency*1000)
        ''''''
    latency_avg = sum(latencies) / len(latencies) 
</pre>


**Model size**: (Unit: Byte)

It is presupposed that the model, whose size is to be calculated, has already undergone quantization, each time with different search option.

For each search option, we calculate the total storage size of the model by iterating through the space occupied by the weights of each layer.
<pre>
The memory footprint of each layer is determined by the following attributes:
Linear: weight, bias
Batchnorm: weight(γ), bias(β), mean, variance
ReLU: None
</pre>

The subsequent script is designed for assessing the memory consumption attributed to the model
<pre>
def model_storage_size(model, weight_bit_width, bias_bit_width, data_bit_width):
    total_bits = 0 
    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name:
            bits = param.numel() * weight_bit_width
            total_bits += bits
        elif param.requires_grad and 'bias' in name:
            bits = param.numel() * bias_bit_width
            total_bits += bits
    total_bits += data_bit_width*(1*16+1) # mean and variance of batchnorm
    total_bytes = total_bits / 8
    return total_bytes

for i, config in enumerate(search_spaces):
    # definition of weight & bias & data width
    size = model_storage_size(mg.model, weight_bit_width, bias_bit_width, data_bit_width)
    ''''''
</pre>


**FLOPs**: (Unit: number)

For each search option, we compute the FLOPs for linear, batchnorm, and relu module.

We employ the same methodology as outlined in the optional task of Lab2. 
(Detailed explanations are also available in the code comments)

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
    input_features = input_features*batch_size
    return input_features
def add_flops_bitops_analysis_pass(graph):
    total_flops = 0
    for node in graph.fx_graph.nodes:
        if isinstance(get_node_actual_target(node), torch.nn.modules.Linear):
            flops = calculate_flop_for_linear(get_node_actual_target(node), batch_size)
            total_flops += flops
        elif isinstance(get_node_actual_target(node), torch.nn.modules.BatchNorm1d):
            flops = calculate_flop_for_batchnorm1d(get_node_actual_target(node), batch_size)
            total_flops += flops
    flops = calculate_flop_for_relu(get_node_actual_target(node), 16, batch_size)
    total_flops += flops
    flops = calculate_flop_for_relu(get_node_actual_target(node), 5, batch_size)
    total_flops += flops
    return total_flops
</pre>


**Bit-wise operations**: (Unit: number)

For each search option, we compute the bitwise operations count for linear module。

We employ identical methodology as outlined in the optional task of Lab2.

<pre>
def bit_wise_op(model, input_res, data_width, weight_width, bias_width, batch_size):
    total_bitwise_ops = 0
    for name, module in model.named_modules():
        if isinstance(module, LinearInteger):
            bitwise_ops = calculate_bitwise_ops_for_linear(module, input_res, data_width, weight_width, bias_width, batch_size)
            total_bitwise_ops += bitwise_ops
    return total_bitwise_ops
def calculate_bitwise_ops_for_linear(module, input_res, data_bit_width, weight_bit_width, bias_bit_width, batch_size):
    in_features = module.in_features
    out_features = module.out_features
    bitwise_ops_per_multiplication = data_bit_width * weight_bit_width
    bitwise_ops_per_addition = data_bit_width * weight_bit_width
    bitwise_ops_per_output_feature = in_features * bitwise_ops_per_multiplication + (in_features - 1) * bitwise_ops_per_addition
    if module.bias is not None:
        bitwise_ops_per_output_feature += bias_bit_width
    total_bitwise_ops = out_features * bitwise_ops_per_output_feature
    return total_bitwise_ops*batch_size

for i, config in enumerate(search_spaces):
    # definition of weight & bias & data width
    bit_op = bit_wise_op(mg.model, (16,), data_bit_width, weight_bit_width, bias_bit_width)
    ''''''
</pre>

## 2. Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric

Note that to leverage the pretrained model (as relying solely on randomly initialized parameters across various models during the search process would make metrics like accuracy become meaningless), it is imperative to preload the "best.ckpt" file.
<pre>
CHECKPOINT_PATH = "/mnt/d/imperial/second_term/adls/new/mase/mase_output/jsc-tiny_classification_jsc_2024-02-05/software/training_ckpts/best.ckpt"
# model definition
model = load_model(load_name=CHECKPOINT_PATH, load_type="pl", model=model)
</pre>

Subsequently, inference is performed for each configuration within the search space.
<pre>
# Essential Code Segment (Extraneous elements omitted)
for i, config in enumerate(search_spaces):
    size = model_storage_size(mg.model, weight_bit_width, bias_bit_width, data_bit_width)  # model size after it has been quantized
    flop = add_flops_bitops_analysis_pass(mg)
    bit_op = bit_wise_op(mg.model, (16,), weight_bit_width, bias_bit_width, data_bit_width, batch_size)
    
    acc_avg, loss_avg = 0, 0;  accs, losses, latencies = [], [], []

    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        start_time = time.time()
        preds = mg.model(xs)
        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)

        acc = metric(preds, ys); accs.append(acc)
        loss = torch.nn.functional.cross_entropy(preds, ys); losses.append(loss)

    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    latency_avg = sum(latencies) / len(latencies) 

    # for this particular element in search space
    recorded_metrics.append({
        ......
    })   
</pre>

<img src="../../imgs/3_2.png" width=1000>

The figure above depicts the distinct metrics observed for varying quantization configurations, from which we could derive two main conclusions:

(Remember for the original model without quantisation, the validation accuracy is 51.30%).

1) Evidently, Post-Training Quantization (PTQ) does not compromise validation accuracy while markedly diminishing the storage requirements for data, weights, and biases. Hence, quantization proves to be beneficial.

2) Generally speaking, as the quantization precision of data, weights, and biases increases (i.e., higher retained precision), the performance of the model improves, as can be shown by the increased accuracy and reduced loss (though not obvious in our case). However, this also impacts other metrics to a certain extent, such as a noticeable increase in the latency required to execute a single batch, an augmentation in model size, and a rise in the number of bitwise operations.

Accuracy and loss actually serve as the same quanlity metric. In fact, accuracy and loss are inversely proportional, implying that as the model's prediction accuracy increases, the prediction loss correspondingly decreases. This relationship is evident from the following formula for accuracy (torcheval.metrics.MulticlassAccuracy):

<img src="../../imgs/3_2_1.png" width=200>

This formula quantifies the number of incorrect predictions within a batch. When we consider its negation, it also serves as a form of loss.

## 3. Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.

Currently, the system employs the TPE sampler from Optuna, which is a Bayesian optimization method.

Alternatively, the BruteForceSampler from Optuna can also be utilized for hyperparameter optimization.

<pre>
# optuna.py within search strategies
def sampler_map(self, name):
    ''''''
    case "brute-force":
        sampler = optuna.samplers.BruteForceSampler()
    return sampler
# jsc_toy_by_type.toml (actually this name should be changed to jsc_tiny_by_type.toml as it uses the JSC-Tiny model)
[search.strategy.setup]
''''''
sampler = "brute-force"
</pre>

It's important to note that due to the way the search space is defined, we represent data_width, data_frac_width, weight_width, and weight_frac_width in the TOML file as follows:
<pre>
[search.search_space.seed.seq_blocks_2.config]  # In this case, we choose by "name" to locate the linear module.
name = ["integer"]
data_in_width = [8, 8, 4, 16]
data_in_frac_width = [4, 6, 2, 8]
weight_width = [8, 8, 4, 16]
weight_frac_width = [4, 6, 2, 8]
bias_width = [8]
bias_frac_width = [4]
''''''
n_trials = 256 # 4*4*4*4
</pre>

Therefore, within our search space, we have 4×4×4×4=256 search options available. This configuration does not strictly align with a one-to-one correspondence to the previously defined search space requirements. However, through these 256 approaches, we can **comprehensively cover** the entire scope of the previously defined search space.

**IMPORTANT**: 

The rationale behind randomizing the order of our search space (e.g., [8, 8, 4, 16] instead of [16, 8, 8, 4]) stems from our objective to evaluate the sample efficiency of two different samplers in the subsequent analysis. If we were to maintain a sequential order, the brute-force method might very likely encounter the configuration with the highest accuracy on its first iteration.

However, this would not accurately reflect its overall sample efficiency. By randomizing the search space, we aim to derive a more general conclusion regarding the samplers' performance in terms of sample efficiency.

Then we execute the command:
<pre>
!./ch search --config configs/examples/jsc_toy_by_type.toml --load /mnt/d/imperial/second_term/adls/new/mase/mase_output/jsc-tiny_classification_jsc_2024-02-05/software/training_ckpts/best.ckpt
</pre>

And we get:

<img src="../../imgs/3_3.png" width=1000>

The trials with the largest accuracy is 52%, indicating that the quantisation has the capability to significantly reduce the model storage space while largely maintain accuracy.

## 4. Comparison between brute-force search and TPE based search regarding sample efficiency.

Sample efficiency refers to the capability of identifying optimal (or near-optimal) hyperparameters utilizing the minimal number of trials ("samples").

Therefore, in the context of evaluating different samplers, we assess their performance based on the accuracy of the best trial in relation to the number of trials conducted.

<pre>
# The best trial accuracy when n_trial=1 (brute-force sampler)
<img src="../../imgs/3_4_bruteforce.png" width=1000>
    
# The best trial accuracy when n_trial=1 (tpe sampler)
<img src="../../imgs/3_4_tpe.png" width=1000>
</pre>

Therefore, as both sampler only have one trial, we can see that the *tpe* sampler has a much higher sample efficiency compared to the *brute-force* sampler.

We could continue to check the best trial accuracy for both samplers from n_trial=1 to 5.

<img src="../../imgs/3_4_compare.png" width=600>

It is evident that the TPE method consistently maintains a relatively high best trial accuracy. In contrast, the brute-force method exhibits significant fluctuations in accuracy as the data_loader order has been randomized.




# Lab4 

## 1. Modify the network to have layers expanded to double sizes.

We will adjust the configuration of each linear layer by applying a channel multiplier factor of 2.

Regarding the ReLU activation layers, as the nn.ReLU module from PyTorch does not require any parameters for its initialization. Therefore, we'll standardize all ReLU layers to nn.ReLU().

<pre>
def instantiate_relu(boolean):
    return nn.ReLU(inplace=boolean)
def redefine_relu_pass(graph, pass_args=None):
    pass_args_copy = copy.deepcopy(pass_args)
    main_config = pass_args_copy.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    for node in graph.fx_graph.nodes:
        i += 1
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        if name is not None:
            new_module = instantiate_relu(True)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
    return graph, {}
    
pass_config_linear = {
"by": "name",
"default": {"config": {"name": None}},
"seq_blocks_2": {
    "config": {
        "name": "output_only",
        "channel_multiplier": 2,
        }
"seq_blocks_4": {
    "config": {
        "name": "both",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_6": {
    "config": {
        "name": "input_only",
        "channel_multiplier": 2,
        }
    },
}
pass_config_relu = {
"by": "name",
"default": {"config": {"name": None}},
"seq_blocks_3": {
    "config": {
        "name": "relu",
        }
    },
"seq_blocks_5": {
    "config": {
        "name": "relu",
        }
    },
}
</pre>

Then, we can obtain the updated model with its layers' dimensions doubled.

<img src="../../imgs/4_1.png" width=800>

## 2. Use grid search to search for the best channel multiplier value.

To ascertain the most effective channel multiplier, we have established a search space designed for selecting the optimal channel multiplier.

<pre>
search_space = [1,2,3,4,5]  # the set where channel multiplier selects its value from
</pre>

**Training Process**: 

Contrary to the approach taken in Lab3 where pretrained models were loaded, the various architecture modifications proposed for Lab4 remain untrained. Therefore, it is imperative to subject these modified networks to a comprehensive training before we perform any search. Otherwise, proceeding directly to inference with the dataloader on these untrained models would result in evaluations that lack substantive value. 

We set max_epoch=10 for training and batch_size=512 for the dataloader.

<pre>
# Essential Code Segment
for multiplier in channel_multiplier:
    # get sampled_config
    # define pass_config_linear & pass_config_relu 
    mg, _ = redefine_linear_transform_pass(graph=mg, pass_args={"config": pass_config_linear})
    mg, _ = redefine_relu_pass(graph=mg, pass_args={"config": pass_config_relu})

    for epoch in range(max_epoch): # sampled_config
        for inputs in data_module.train_dataloader():
            xs, ys = inputs
            optimizer.zero_grad()
            preds = mg.model(xs)
            loss = torch.nn.functional.cross_entropy(preds, ys)  
            loss.backward()  
            optimizer.step() 

    # save model, initialize masegraph and optimizer again
</pre>

Subsequently, we executed the search.

Given the network's simplicity, indicating manageable model size and reasonable latency, we focused exclusively on two key performance metrics **accuracy and loss** for evaluation.
<pre>
for multiplier in channel_multiplier:
    # get sampled_config
    # define pass_config_linear & pass_config_relu 
    mg, _ = redefine_linear_transform_pass(graph=mg, pass_args={"config": pass_config_linear})
    mg, _ = redefine_relu_pass(graph=mg, pass_args={"config": pass_config_relu})

    mymodel = load_model(f"mase_output/4_2/model_with_multiplier_{multiplier}.ckpt", "pl", mg.model) # load pre-trained model

    acc_avg, loss_avg = 0, 0 ; accs, losses = [], []
    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        preds = mymodel(xs)
        acc = metric(preds, ys)
        accs.append(acc)
        loss = torch.nn.functional.cross_entropy(preds, ys)
        losses.append(loss)
    
    acc_avg = sum(accs) / len(accs) ; loss_avg = sum(losses) / len(losses)
    recorded_accs.append({"acc":acc_avg,"loss":loss_avg})
</pre>

Then, we can obtain the accuracy and loss of the models corresponding to each channel multiplier value.

<img src="../../imgs/4_2.png" width=200>

In our case, the best model is the one with multiplier=5, with an accuracy of 23.5%.

## 3. Search for Optimal Channel Multipliers with Independent Layer Scaling

To achieve individual scaling of each layer rather than uniform scaling across the network, we assign distinct channel multiplier values to each adjustable channel input/output.

We establish a search space defined by the set [1, 2, 3, 4, 5], where channel multipliers are selected, allowing for customized scaling of the network's layers.

For each point of channel input/output modification, unique multiplier variables are designated, namely, **a, b, c, d**. These variables individually adjust the scaling factor applied to their corresponding channel dimensions.

<img src="../../imgs/4_3.png" width=500>

Please note that we will actually only iterate through values for a and c. as b=a and d=c for the consistency of channel output and the next channel's input.

The following is the impelmentation for search space, the new *pass_config* method, and the updated *linear_transform_pass* function.
<pre>
# Essential Code Segment
# within main function
search_space = [1,2,3,4,5]

# within design_pass_config_linear(a,b,c,d):
"seq_blocks_2": {
            "config": {
                "name": "output_only",
                "channel_multiplier_output": a,
            }
        },
        "seq_blocks_4": {
            "config": {
                "name": "both",
                "channel_multiplier_input": b,
                "channel_multiplier_output": c,
            }
        },
        "seq_blocks_6": {
            "config": {
                "name": "input_only",
                "channel_multiplier_input": d,
            }
        },

# within redefine_linear_transform_pass(graph, pass_args=None):  
if name == "output_only":
    out_features = out_features * config["channel_multiplier_output"] 
elif name == "both":
    in_features = in_features * config["channel_multiplier_input"] 
    out_features = out_features * config["channel_multiplier_output"] 
elif name == "input_only":
    in_features = in_features * config["channel_multiplier_input"] 
</pre>

We first perform training on different models. As with before, we set max_epoch=10 and batch_size=512.
<pre>
# Essential Code Segment
multipliers = [1, 2, 3, 4, 5]
max_epoch=10
batch_size = 512
optimizer = optim.Adam(model.parameters(), lr=1e-5)

for a in multipliers:
    for c in multipliers:
        b = a ; d = c
        pass_config_linear = design_pass_config_linear(a, b, c, d)

        mg = init_mg()
        mg, _ = redefine_linear_transform_pass(mg, pass_args={"config": pass_config_linear})
        mg, _ = redefine_relu_pass(mg, pass_args={"config": pass_config_relu})

        for epoch in range(max_epoch):
            for inputs in data_module.train_dataloader():
                xs, ys = inputs
                optimizer.zero_grad()
                preds = mg.model(xs)
                loss = torch.nn.functional.cross_entropy(preds, ys)  
                loss.backward()  
                optimizer.step() 

        '''''' model save & updates for mg and optimizer
</pre>

Subsequently, we executed the search.

<pre>
multipliers = [1, 2, 3, 4, 5]
recorded_metrics = []
metric = MulticlassAccuracy(num_classes=5)

for a in multipliers:
    for c in multipliers:
        b = a ; d = c
        pass_config_linear = design_pass_config_linear(a, b, c, d)

        mg = init_mg()
        mg, _ = redefine_linear_transform_pass(mg, pass_args={"config": pass_config_linear})
        mg, _ = redefine_relu_pass(mg, pass_args={"config": pass_config_relu})

        mymodel = load_model(f"mase_output/4_3/model_with_multiplier_{a}_{b}_{c}_{d}.ckpt", "pl", mg.model)

        acc_avg, loss_avg = 0, 0
        accs, losses = [], []

        with torch.no_grad():
            for inputs in data_module.train_dataloader():
                xs, ys = inputs
                preds = mg.model(xs)
                acc = metric(preds, ys)
                accs.append(acc.item())
                loss = torch.nn.functional.cross_entropy(preds, ys)
                losses.append(loss.item())

        acc_avg = sum(accs) / len(accs)
        loss_avg = sum(losses) / len(losses)
        recorded_metrics.append({"block2_output":a, "block4_input":b, "block4_output":c, "block6_input":d, "acc(%)":acc_avg*100, "loss":loss_avg})
</pre>

Then we obtain the result:

<img src="../../imgs/4_3_2.png" width=600>

We find that when a=b=1 and c=d=4, the model has the highest accuracy of 32.2%, which corresponds to the following model:

<img src="../../imgs/4_3_3.png" width=600>


## 4. Integrate the search to the chop flow, so we can run it from the command line.

I create my own class of <code>NetworkArchitectureSearch</code> inherited from SearchSpaceBase at <code>search.search_space.quantization.network_architecture.py</code>. 

Within the class, I mainly define four functions:

1. _post_init_setup(self)

I inherit this from <code>base.py</code>.

It primarily determines the computing resource, and initializes some variables, such as masegraph and default network config of a,b,c,d.

2. rebuild_model(self, sampled_config, is_eval_mode: bool = True)

Each time when the configuration updates, we need to rebuild the model with specific sampled_config.

It mainly consists of three steps:

2.1 Initialisation:
<pre>
mg = MaseGraph(self.model) # only pass the architecture but not the weights!
mg,_ = init_metadata_analysis_pass(mg, None)
</pre>

2.2 Change the model according to the sampled_config:
<pre>
mg, _ = self.redefine_linear_transform_pass(mg, sampled_config)
mg, _ = self.redefine_relu_pass(mg, sampled_config)
</pre>

2.3 Load the pretrained model with that sampled_config:
<pre>
mymodel = load_model(f"mase_output/4_3/model_with_multiplier_{numbers[0]}_{numbers[1]}_{numbers[2]}_{numbers[3]}.ckpt", "pl", mg.model)
</pre>
We must load the pre-trained models in that proceeding directly to inference with the dataloader on the untrained models would result in evaluations that are meaningless.

Finally, we return the masegraph with specified sampled_config.

3 build_search_space

Build the search space for different channel multiplier input/output.
<pre>
multiplier_options = [1,2,3,4,5]
self.choices_flattened = {
    "seq_blocks_2_channel_multiplier_output": multiplier_options, 
    "seq_blocks_4_channel_multiplier_output": multiplier_options,
}
self.choice_lengths_flattened = {k: len(v) for k, v in self.choices_flattened.items()}
</pre>

4 flattened_indexes_to_config(self, indexes: dict[str, int]):

Originally, configs are presented in such a way:
<pre>
"seq_blocks_2_channel_multiplier_output": 2,
"seq_blocks_4_channel_multiplier_output": 3,
</pre>

Now we need to reshape it into the ready-to-use type:
<pre>
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
</pre>

Note that we also need to define <code>DEFAULT_NETWORK_CONFIG</code>, as well as the previously defined functions for modifying network structure based on sampled_config, including <code>redefine_linear_transform_pass</code> and <code>redefine_relu_pass</code>.

The appropriate import statements must be implemented.
<pre>
# search.search_space.quantization.__init__.py
from .network_architecture import NetworkArchitectureSearch
# search.search_space.__init__.py
from .quantization import NetworkArchitectureSearch
SEARCH_SPACE_MAP = {"graph/quantize/network_search": NetworkArchitectureSearch,}
</pre>

In addition to establishing a custom search space, we must also generate a bespoke TOML file.

We need to specify our task, together with the new search space within the file.

<pre>
# Only essential Code Segment are presented
model = "jsc-three-linear-layers"
dataset = "jsc"
task = "cls"
''''''
[search.search_space]
name = "graph/quantize/network_search"
[search.search_space.setup]
by = "name"
[search.search_space.seed.seq_blocks_2.config]
channel_multiplier_output = [1, 2, 3, 4, 5]
[search.search_space.seed.seq_blocks_4.config]
channel_multiplier_input = [1, 2, 3, 4, 5]
channel_multiplier_output = [1, 2, 3, 4, 5]
[search.search_space.seed.seq_blocks_6.config]
channel_multiplier_input = [1, 2, 3, 4, 5]
''''''
[search.strategy.setup]
n_trials = 25  # altogether 5*5=25 search options within the search space
</pre>

Finally, we execute the command:
<pre>
!./ch search --config configs/examples/jsc_network_search.toml --load /mnt/d/imperial/second_term/adls/new/mase/mase_output/jsc-three-linear-layers_classification_jsc_2024-01-31/software/training_ckpts/best.ckpt
</pre>

Then we found that the search option with the highest accuracy achieved an accuracy rate of 24.2%.
<img src="../../imgs/4_4.png" width=1000>


## Optional Task (scaling the search to real networks)

As the methodology and approach are fundamentally similar to the previous question, and considering the substantial volume of code, we have refrained from including specific code snippets here. FUll code are provided here: [Full Code](https://github.com/JasonShen-SH/mase_real/tree/main)

To start with, we have modified the presentation format of the VGG model, transforming it into a sequence of network layers to simplify the process of writing the network configuration (making it much easier to modify the network architecture).

The modified version of VGG model will only have <code>self.seq_blocks</code> in the forward function.
<pre>
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.seq_blocks(x)
</pre>

After modifying the model representation, we need to define our search space, which in this instance is [64, 128, 256, 512]. This search space is designated for the output of all convolutional layers, meaning the output channels of all convolutional layers can only be selected from these numbers.
<pre>
# part of build_search_space:
search_space = [64, 128, 256, 512]
self.choices_flattened = {
    "seq_blocks_0_channel_output": search_space,
    "seq_blocks_3_channel_output": search_space,
    "seq_blocks_7_channel_output": search_space,
    "seq_blocks_10_channel_output": search_space,
    "seq_blocks_14_channel_output": search_space,
    "seq_blocks_17_channel_output": search_space,
}
self.choice_lengths_flattened = {k: len(v) for k, v in self.choices_flattened.items()}
</pre>

(Notice1): Additionally, we have stipulated that the number of channels must be ensured to be at least **non-decreasing**, meaning the output channel count of a subsequent convolutional layer must not be less than that of its preceding layer. 

(Notice2&3): Naturally, the input channel count of each convolutional layer must match the output channel count of the preceding convolutional layer, and the channel count for batch normalization must also correspond with the preceding convolutional layer.

(Notice4): Lastly, it is imperative to ensure that the first linear layer following the flattening operation aligns with the feature dimension. 

We will articulate the aforementioned four points as four separate notices. Below is their code implementation:

<pre>
# part of flattened_indexes_to_config:
    
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
</pre>


Concurrently, we need to introduce new transformations for newly-introduced modules such as conv2d, bn and pooling (i.e., conv2d_transform, bn_transform and pooling_transform modules):
<pre>
# Only essential Code Segment are presented
# for conv2d:
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
    
# for bn:
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
    
# for pooling_transform:
 if name == "maxpool":
    ori_module = graph.modules[node.target]
    if isinstance(ori_module, nn.MaxPool2d):
        new_module = self.instantiate_maxpool()
        parent_name, name = get_parent_name(node.target)
        setattr(graph.modules[parent_name], name, new_module)
</pre>

We must also make corresponding modifications to the .toml file, which in this case, I have named <code>cifar10_vgg.toml</code>.

Ultimately, we execute the following command to initiate the search operation:
<pre>
!./ch search --config configs/examples/cifar10_vgg.toml --load ../mase_output/vgg7_classification_cifar10_2024-02-01/software/training_ckpts/best.ckpt
</pre>

In reality, we should complete the pre-training process for models under every configuration (for instance, training each search option for 10 epochs as we did previously). 

However, due to the number of models with different configurations and GPU resource constraints, we proceed directly to inference, which might result in relatively limited accuracy.

<pre>
Best trial(s):
|    |   number | software_metrics                  | hardware_metrics                                | scaled_metrics                              |
|----+----------+-----------------------------------+-------------------------------------------------+---------------------------------------------|
|  0 |        0 | {'loss': 2.309, 'accuracy': 0.11} | {'average_bitwidth': 32, 'memory_density': 1.0} | {'accuracy': 0.11, 'average_bitwidth': 6.4} |
</pre>







