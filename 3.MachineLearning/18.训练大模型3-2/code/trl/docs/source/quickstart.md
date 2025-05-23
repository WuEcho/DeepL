# Quickstart

## How does it work?

Fine-tuning a language model via PPO consists of roughly three steps:

1. **Rollout**: The language model generates a response or continuation based on a query which could be the start of a sentence.
2. **Evaluation**: The query and response are evaluated with a function, model, human feedback, or some combination of them. The important thing is that this process should yield a scalar value for each query/response pair. The optimization will aim at maximizing this value.
3. **Optimization**: This is the most complex part. In the optimisation step the query/response pairs are used to calculate the log-probabilities of the tokens in the sequences. This is done with the model that is trained and a reference model, which is usually the pre-trained model before fine-tuning. The KL-divergence between the two outputs is used as an additional reward signal to make sure the generated responses don't deviate too far from the reference language model. The active language model is then trained with PPO.

The full process is illustrated in the following figure:
<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl_overview.png"/>

## Minimal example

The following code illustrates the steps above. 

```python
# 0. imports
import torch
from transformers import GPT2Tokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


# 1. load a pretrained model
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. initialize trainer
ppo_config = {"mini_batch_size": 1, "batch_size": 1}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

# 3. encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.pretrained_model.device)

# 4. generate model response
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 20,
}
response_tensor = ppo_trainer.generate([item for item in query_tensor], return_prompt=False, **generation_kwargs)
response_txt = tokenizer.decode(response_tensor[0])

# 5. define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0, device=model.pretrained_model.device)]

# 6. train model with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
```

In general, you would run steps 3-6 in a for-loop and run it on many diverse queries. You can find more realistic examples in the examples section. 

## How to use a trained model

After training a `AutoModelForCausalLMWithValueHead`, you can directly use the model in `transformers`.
```python

# .. Let's assume we have a trained model using `PPOTrainer` and `AutoModelForCausalLMWithValueHead`

# push the model on the Hub
model.push_to_hub("my-fine-tuned-model-ppo")

# or save it locally
model.save_pretrained("my-fine-tuned-model-ppo")

# load the model from the Hub
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("my-fine-tuned-model-ppo")
```

You can also load your model with `AutoModelForCausalLMWithValueHead` if you want to use the value head, for example to continue training.

```python
from trl.model import AutoModelForCausalLMWithValueHead

model = AutoModelForCausalLMWithValueHead.from_pretrained("my-fine-tuned-model-ppo")
```
