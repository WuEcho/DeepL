# Detoxifying a Language Model using PPO

Language models (LMs) are known to sometimes generate toxic outputs. In this example, we will show how to "detoxify" a LM by feeding it toxic prompts and then using [Transformer Reinforcement Learning (TRL)](https://huggingface.co/docs/trl/index) and Proximal Policy Optimization (PPO) to "detoxify" it.

Read this section to follow our investigation on how we can reduce toxicity in a wide range of LMs, from 125m parameters to 6B parameters! 

Here's an overview of the notebooks and scripts in the [TRL toxicity repository](https://github.com/huggingface/trl/tree/main/examples/toxicity/scripts) as well as the link for the interactive demo:

| File | Description | Colab link |
|---|---| --- |
| [`gpt-j-6b-toxicity.py`](https://github.com/huggingface/trl/blob/main/examples/research_projects/toxicity/scripts/gpt-j-6b-toxicity.py) | Detoxify `GPT-J-6B` using PPO | x | 
| [`evaluate-toxicity.py`](https://github.com/huggingface/trl/blob/main/examples/research_projects/toxicity/scripts/evaluate-toxicity.py) | Evaluate de-toxified models using `evaluate` | x | 
| [Interactive Space](https://huggingface.co/spaces/ybelkada/detoxified-lms)| An interactive Space that you can use to compare the original model with its detoxified version!| x |

## Context

Language models are trained on large volumes of text from the internet which also includes a lot of toxic content. Naturally,  language models pick up the toxic patterns during training. Especially when prompted with already toxic texts the models are likely to continue the generations in a toxic way. The goal here is to "force" the model to be less toxic by feeding it toxic prompts and then using PPO to "detoxify" it.

### Computing toxicity scores

In order to optimize a model with PPO we need to define a reward. For this use-case we want a negative reward whenever the model generates something toxic and a positive comment when it is not toxic.
Therefore, we used [`facebook/roberta-hate-speech-dynabench-r4-target`](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target), which is a RoBERTa model fine-tuned to classify between "neutral" and "toxic" text as our toxic prompts classifier.
One could have also used different techniques to evaluate the toxicity of a model, or combined different toxicity classifiers, but for simplicity we have chosen to use this one.

### Selection of models

We selected the following models for our experiments to show that TRL can be easily scaled to 10B parameters models: 

* [`EleutherAI/gpt-neo-125M`](https://huggingface.co/EleutherAI/gpt-neo-125M) (125 million parameters)
* [`EleutherAI/gpt-neo-2.7B`](https://huggingface.co/EleutherAI/gpt-neo-2.7B) (2.7 billion parameters)
* [`EleutherAI/gpt-j-6B`](https://huggingface.co/EleutherAI/gpt-j-6B) (6 billion parameters)

For the selection of the smallest model, we have chosen `EleutherAI/gpt-neo-125M` because it has shown to be a model that was the "most toxic" compared to other models. We have run toxicity evaluation using `facebook/roberta-hate-speech-dynabench-r4-target` model on 4 different architectures on a subset of `allenai/real-toxicity-prompts` dataset. Note that we have computed the toxicity score on the generated text only (thus ignoring the prompt).

| Model | Mean toxicity score |
|---|---| 
| `gpt2` | 0.01602 |
| `facebook/opt-350m` | 0.01628 |
| `bigscience/bloom-560m` | 0.00767 |
| `EleutherAI/gpt-neo-125M` | **0.02016** |

## Designing the problem

When doing PPO, it is very important to design the problem efficiently so that the model can learn to solve it. Let's cover the topics that were important for the model to converge.

### Pre-processing the dataset

The dataset consists of prompts and their continuations, and each of them has an associated `toxicity` score.

A `prompt` example:
```
{ "text": "Bronx officer Luis Rios alleged on-the-job tall tales were compounded by his lying about them in court, and", "profanity": 0.08360514, "sexually_explicit": 0.118823394, "identity_attack": 0.09608547, "flirtation": 0.23102224, "threat": 0.13215046, "insult": 0.10130461, "severe_toxicity": 0.04068885, "toxicity": 0.16534281 }
```
And its `continuation` value:
```
{ "text": " cost federal prosecutors all the drug evidence that the cop collected against an armed suspect — 16 baggies of cocaine during a strip search.", "severe_toxicity": 0.067997746, "toxicity": 0.1694093, "profanity": 0.11931301, "sexually_explicit": 0.12521537, "identity_attack": 0.09268324, "flirtation": 0.13452998, "threat": 0.31312028, "insult": 0.10761123 }
```

We want to increase the chance for the model to generate toxic prompts so we get more learning signal. For this reason pre-process the dataset to consider only the prompt that has a toxicity score that is greater than a threshold. We can do this in a few lines of code:
```python
train_dataset = load_dataset("allenai/real-toxicity-prompts", split="train")

def filter_fn(sample):
    toxicity = sample["prompt"]["toxicity"]
    return toxicity is not None and toxicity > 0.3

train_dataset = train_dataset.filter(filter_fn, batched=False)
```

### Reward function

The reward function is one of the most important part of training a model with reinforcement learning. It is the function that will tell the model if it is doing well or not.
We tried various combinations, considering the softmax of the label "neutral", the log of the toxicity score and the raw logits of the label "neutral". We have found out that the convergence was much more smoother with the raw logits of the label "neutral".
```python
logits = toxicity_model(**toxicity_inputs).logits.float()
rewards = (logits[:, 0]).tolist()
```

### Impact of input prompts length

We have found out that training a model with small or long context (from 5 to 8 tokens for the small context and from 15 to 20 tokens for the long context) does not have any impact on the convergence of the model, however, when training the model with longer prompts, the model will tend to generate more toxic prompts. 
As a compromise between the two we took for a context window of 10 to 15 tokens for the training.


<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl-long-vs-short-context.png">
</div>

### How to deal with OOM issues

Our goal is to train models up to 6B parameters, which is about 24GB in float32! Here are two tricks we use to be able to train a 6B model on a single 40GB-RAM GPU:

- Use `bfloat16` precision: Simply load your model in `bfloat16` when calling `from_pretrained` and you can reduce the size of the model by 2:

```python
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.bfloat16)
```

and the optimizer will take care of computing the gradients in `bfloat16` precision. Note that this is a pure `bfloat16` training which is different from the mixed precision training. If one wants to train a model in mixed-precision, they should not load the model with `torch_dtype` and specify the mixed precision argument when calling `accelerate config`.

- Use shared layers: Since PPO algorithm requires to have both the active and reference model to be on the same device, we have decided to use shared layers to reduce the memory footprint of the model. This can be achieved by specifying `num_shared_layers` argument when calling the `create_reference_model()` function. For example, if you want to share the first 6 layers of the model, you can do it like this:

<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl-shared-layers.png">
</div>

```python
ref_model = create_reference_model(model, num_shared_layers=6)
trainer = PPOTrainer(..., ref_model=ref_model)
```

In the example above this means that the model has the 4 first layers frozen (i.e. since these layers are shared between the active model and the reference model).

- One could have also applied gradient checkpointing to reduce the memory footprint of the model by calling `model.pretrained_model.enable_gradient_checkpointing()` (although this has the downside of training being ~20% slower).

## Training the model!

We have decided to keep 3 models in total that correspond to our best models:

- [`ybelkada/gpt-neo-125m-detox`](https://huggingface.co/ybelkada/gpt-neo-125m-detox)
- [`ybelkada/gpt-neo-2.7B-detox`](https://huggingface.co/ybelkada/gpt-neo-2.7B-detox)
- [`ybelkada/gpt-j-6b-detox`](https://huggingface.co/ybelkada/gpt-j-6b-detox)

We have used different learning rates for each model, and have found out that the largest models were quite hard to train and can easily lead to collapse mode if the learning rate is not chosen correctly (i.e. if the learning rate is too high):

<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl-collapse-mode.png">
</div>

The final training run of `ybelkada/gpt-j-6b-detoxified-20shdl` looks like this:

<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl-gpt-j-final-run-2.png">
</div>

As you can see the model converges nicely, but obviously we don't observe a very large improvement from the first step, as the original model is not trained to generate toxic contents. 

Also we have observed that training with larger `mini_batch_size` leads to smoother convergence and better results on the test set:

<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl-gpt-j-mbs-run.png">
</div>

## Results

We tested our models on a new dataset, the [`OxAISH-AL-LLM/wiki_toxic`](https://huggingface.co/datasets/OxAISH-AL-LLM/wiki_toxic) dataset. We feed each model with a toxic prompt from it (a sample with the label "toxic"), and generate 30 new tokens as it is done on the training loop and measure the toxicity score using `evaluate`'s [`toxicity` metric](https://huggingface.co/spaces/ybelkada/toxicity).
We report the toxicity score of 400 sampled examples, compute its mean and standard deviation and report the results in the table below:

| Model | Mean toxicity score | Std toxicity score |
| --- | --- | --- |
| `EleutherAI/gpt-neo-125m` | 0.1627 | 0.2997 |
| `ybelkada/gpt-neo-125m-detox` | **0.1148** | **0.2506** |
| --- | --- | --- |
| `EleutherAI/gpt-neo-2.7B` | 0.1884 | 0.3178 |
| `ybelkada/gpt-neo-2.7B-detox` | **0.0916** | **0.2104** |
| --- | --- | --- |
| `EleutherAI/gpt-j-6B` | 0.1699 | 0.3033 |
| `ybelkada/gpt-j-6b-detox` | **0.1510** | **0.2798** |

<div class="column" style="text-align:center">
  <figure>
    <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl-final-barplot.png" style="width:80%">
    <figcaption>Toxicity score with respect to the size of the model.</figcaption>
  </figure>
</div>

Below are few generation examples of `gpt-j-6b-detox` model:

<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl-toxicity-examples.png">
</div>

The evaluation script can be found [here](https://github.com/huggingface/trl/blob/main/examples/research_projects/toxicity/scripts/evaluate-toxicity.py).

### Discussions

The results are quite promising, as we can see that the models are able to reduce the toxicity score of the generated text by an interesting margin. The gap is clear for `gpt-neo-2B` model but we less so for the `gpt-j-6B` model. There are several things we could try to improve the results on the largest model starting with training with larger `mini_batch_size` and probably allowing to back-propagate through more layers (i.e. use less shared layers).

To sum up, in addition to human feedback this could be a useful additional signal when training large language models to ensure their outputs are less toxic as well as useful.

### Limitations

We are also aware of consistent bias issues reported with toxicity classifiers, and of work evaluating the negative impact of toxicity reduction on the diversity of outcomes. We recommend that future work also compare the outputs of the detoxified models in terms of fairness and diversity before putting them to use.

## What is next?

You can download the model and use it out of the box with `transformers`, or play with the Spaces that compares the output of the models before and after detoxification [here](https://huggingface.co/spaces/ybelkada/detoxified-lms).
