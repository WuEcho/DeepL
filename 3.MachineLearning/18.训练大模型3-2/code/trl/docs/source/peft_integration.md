# Examples of using peft with trl to finetune 8-bit models with Low Rank Adaption (LoRA)

The notebooks and scripts in this examples show how to use Low Rank Adaptation (LoRA) to fine-tune models in a memory efficient manner. Most of PEFT methods supported in peft library but note that some PEFT methods such as Prompt tuning are not supported.
For more information on LoRA, see the [original paper](https://huggingface.co/papers/2106.09685).

Here's an overview of the `peft`-enabled notebooks and scripts in the [trl repository](https://github.com/huggingface/trl/tree/main/examples):

| File | Task | Description | Colab link |
|---|---| --- |
| [`stack_llama/rl_training.py`](https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/rl_training.py) | RLHF | Distributed fine-tuning of the 7b parameter LLaMA models with a learned reward model and `peft`. |  |
| [`stack_llama/reward_modeling.py`](https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/reward_modeling.py) | Reward Modeling | Distributed training of the 7b parameter LLaMA reward model with `peft`. |  |
| [`stack_llama/supervised_finetuning.py`](https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py) | SFT | Distributed instruction/supervised fine-tuning of the 7b parameter LLaMA model with `peft`. |  |

## Installation
Note: peft is in active development, so we install directly from their Github page.
Peft also relies on the latest version of transformers. 

```bash
pip install trl[peft]
pip install bitsandbytes loralib
pip install git+https://github.com/huggingface/transformers.git@main
#optional: wandb
pip install wandb
```

Note: if you don't want to log with `wandb` remove `log_with="wandb"` in the scripts/notebooks. You can also replace it with your favourite experiment tracker that's [supported by `accelerate`](https://huggingface.co/docs/accelerate/usage_guides/tracking).

## How to use it?

Simply declare a `PeftConfig` object in your script and pass it through `.from_pretrained` to load the TRL+PEFT model. 

```python
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead

model_id = "edbeeching/gpt-neo-125M-imdb"
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_id, 
    peft_config=lora_config,
)
```
And if you want to load your model in 8bit precision:
```python
pretrained_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name, 
    load_in_8bit=True,
    peft_config=lora_config,
)
```
... or in 4bit precision:
```python
pretrained_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name, 
    peft_config=lora_config,
    load_in_4bit=True,
)
```


## Launch scripts

The `trl` library is powered by `accelerate`. As such it is best to configure and launch trainings with the following commands:

```bash
accelerate config # will prompt you to define the training configuration
accelerate launch examples/scripts/ppo.py --use_peft # launch`es training
```

## Using `trl` + `peft` and Data Parallelism

You can scale up to as many GPUs as you want, as long as you are able to fit the training process in a single device. The only tweak you need to apply is to load the model as follows:
```python
from peft import LoraConfig
...

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

pretrained_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name, 
    peft_config=lora_config,
)
```
And if you want to load your model in 8bit precision:
```python
pretrained_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name, 
    peft_config=lora_config,
    load_in_8bit=True,
)
```
... or in 4bit precision:
```python
pretrained_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name, 
    peft_config=lora_config,
    load_in_4bit=True,
)
```
Finally, make sure that the rewards are computed on correct device as well, for that you can use `ppo_trainer.model.current_device`.

## Naive pipeline parallelism (NPP) for large models (>60B models)

The `trl` library also supports naive pipeline parallelism (NPP) for large models (>60B models). This is a simple way to parallelize the model across multiple GPUs. 
This paradigm, termed as "Naive Pipeline Parallelism" (NPP) is a simple way to parallelize the model across multiple GPUs. We load the model and the adapters across multiple GPUs and the activations and gradients will be naively communicated across the GPUs. This supports `int8` models as well as other `dtype` models.

<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl-npp.png">
</div>

### How to use NPP?

Simply load your model with a custom `device_map` argument on the `from_pretrained` to split your model across multiple devices. Check out this [nice tutorial](https://github.com/huggingface/blog/blob/main/accelerate-large-models.md) on how to properly create a `device_map` for your model. 
 
Also make sure to have the `lm_head` module on the first GPU device as it may throw an error if it is not on the first device. As this time of writing, you need to install the `main` branch of `accelerate`: `pip install git+https://github.com/huggingface/accelerate.git@main` and `peft`: `pip install git+https://github.com/huggingface/peft.git@main`.

### Launch scripts

Although `trl` library is powered by `accelerate`, you should run your training script in a single process. Note that we do not support Data Parallelism together with NPP yet.

```bash
python PATH_TO_SCRIPT
```

## Fine-tuning Llama-2 model

You can easily fine-tune Llama2 model using `SFTTrainer` and the official script! For example to fine-tune llama2-7b on the Guanaco dataset, run (tested on a single NVIDIA T4-16GB):

```bash
python trl/scripts/sft.py --output_dir sft_openassistant-guanaco  --model_name meta-llama/Llama-2-7b-hf --dataset_name timdettmers/openassistant-guanaco --load_in_4bit --use_peft --per_device_train_batch_size 4 --gradient_accumulation_steps 2
```
