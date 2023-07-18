"""
Check the doc at mase-tools/docs/large_language_models/accelerate_fsdp.md
"""

import os
import sys

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "machop"
    )
)

from chop.actions.accelerate_train import train
from chop.dataset import MyDataModule

# from chop.models.manual.llama_plain.modeling_llama import LlamaForCausalLM
# from transformers.models.llama import LlamaTokenizer
from chop.models.manual.opt_plain.modeling_opt import OPTForCausalLM
from transformers import AutoTokenizer


def main():
    model_name = "facebook/opt-350m"
    task = "lm"
    dataset_name = "wikitext2"
    max_token_len = 128
    batch_size = 1
    num_workers = os.cpu_count()
    optimizer = "adamw"
    max_epochs: int = 1
    max_steps: int = 1
    gradient_accumulation_steps: int = 1
    # Reduced for unit test
    # max_epochs: int = 2
    # max_steps: int = -1
    # gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    lr_scheduler_type: str = "linear"
    num_warmup_steps: int = 0
    save_path: str = "./ckpts"
    load_name: str = None
    load_type: str = ""

    model = OPTForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data_module = MyDataModule(
        model_name=None,
        dataset_name=dataset_name,
        batch_size=batch_size,
        workers=num_workers,
        tokenizer=tokenizer,
        max_token_len=max_token_len,
    )

    train(
        model=model,
        task=task,
        data_module=data_module,
        optimizer=optimizer,
        max_epochs=max_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type=lr_scheduler_type,
        num_warmup_steps=num_warmup_steps,
        save_path=save_path,
        load_name=load_name,
        load_type=load_type,
    )


if __name__ == "__main__":
    main()
