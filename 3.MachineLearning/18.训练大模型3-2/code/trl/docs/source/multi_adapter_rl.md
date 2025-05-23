# Multi Adapter RL (MARL) - a single base model for everything

Here we present an approach that uses a single base model for the entire PPO algorithm - which includes retrieving the reference logits, computing the active logits and the rewards. This feature is experimental as we did not test the convergence of the approach. We encourage the community to let us know if they potentially face issues.

## Requirements

You just need to install `peft` and optionally install `bitsandbytes` as well if you want to go for 8bit base models, for more memory efficient finetuning.

## Summary

You need to address this approach in three stages that we summarize as follows:

1- Train a base model on the target domain (e.g. [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb)) - this is the Supervised Fine Tuning stage - it can leverage the `SFTTrainer` from TRL.
2- Train a reward model using `peft`. This is required in order to re-use the adapter during the RL optimisation process (step 3 below). We show an example of leveraging the `RewardTrainer` from TRL in [this example](https://github.com/huggingface/trl/tree/main/examples/scripts/reward_modeling.py)
3- Fine tune new adapters on the base model using PPO and the reward adapter. ("0 abstraction RL")

Make sure to use the same model (i.e. same architecture and same weights) for the stages 2 & 3. 

## Quickstart

Let us assume you have trained your reward adapter on `llama-7b` model using `RewardTrainer` and pushed the weights on the hub under `trl-lib/llama-7b-hh-rm-adapter`. 
When doing PPO, before passing the model to `PPOTrainer` create your model as follows:

```python
model_name = "huggyllama/llama-7b"
rm_adapter_id = "trl-lib/llama-7b-hh-rm-adapter"

# PPO adapter
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    peft_config=lora_config,
    reward_adapter=rm_adapter_id,
)

...
trainer = PPOTrainer(
    model=model,
    ...
)

...
```
Then inside your PPO training loop, call the `compute_reward_score` method by accessing the `model` attribute from `PPOTrainer`.

```python
rewards = trainer.model.compute_reward_score(**inputs)
```

## Advanced usage

### Control on the adapter name 

If you are familiar with the `peft` library, you know that you can use multiple adapters inside the same model. What you can do is train multiple adapters on the same base model to fine-tune on different policies. 
In this case, you want to be able to control the adapter name you want to activate back, after retrieving the reward. For that, simply pass the appropriate `adapter_name` to `ppo_adapter_name` argument when calling `compute_reward_score`.

```python
adapter_name_policy_1 = "policy_1"
rewards = trainer.model.compute_reward_score(**inputs, ppo_adapter_name=adapter_name_policy_1)
...
```

### Using 4-bit and 8-bit base models

For more memory efficient fine-tuning, you can load your base model in 8-bit or 4-bit while keeping the adapters in the default precision (float32).
Just pass the appropriate arguments (i.e. `load_in_8bit=True` or `load_in_4bit=True`) to `AutoModelForCausalLMWithValueHead.from_pretrained` as follows (assuming you have installed `bitsandbytes`):
```python
model_name = "llama-7b"
rm_adapter_id = "trl-lib/llama-7b-hh-rm-adapter"

# PPO adapter
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    peft_config=lora_config,
    reward_adapter=rm_adapter_id,
    load_in_8bit=True,
)

...
trainer = PPOTrainer(
    model=model,
    ...
)
...
```
