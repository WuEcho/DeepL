# Denoising Diffusion Policy Optimization

[![](https://img.shields.io/badge/All_models-DDPO-blue)](https://huggingface.co/models?other=ddpo,trl)

## The why

| Before | After DDPO finetuning |
| --- | --- |
| <div style="text-align: center"><img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/pre_squirrel.png"/></div> |  <div style="text-align: center"><img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/post_squirrel.png"/></div> |
| <div style="text-align: center"><img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/pre_crab.png"/></div> |  <div style="text-align: center"><img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/post_crab.png"/></div> |
| <div style="text-align: center"><img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/pre_starfish.png"/></div> |  <div style="text-align: center"><img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/post_starfish.png"/></div> |


## Getting started with Stable Diffusion finetuning with reinforcement learning

The machinery for finetuning of Stable Diffusion models with reinforcement learning makes heavy use of HuggingFace's `diffusers`
library. A reason for stating this is that getting started requires a bit of familiarity with the `diffusers` library concepts, mainly two of them - pipelines and schedulers.
Right out of the box (`diffusers` library), there isn't a `Pipeline` nor a `Scheduler` instance that is suitable for finetuning with reinforcement learning. Some adjustments need to be made. 

There is a pipeline interface that is provided by this library that is required to be implemented to be used with the `DDPOTrainer`, which is the main machinery for fine-tuning Stable Diffusion with reinforcement learning. **Note: Only the StableDiffusion architecture is supported at this point.**
There is a default implementation of this interface that you can use out of the box. Assuming the default implementation is sufficient and/or to get things moving, refer to the training example alongside this guide. 

The point of the interface is to fuse the pipeline and the scheduler into one object which allows for minimalness in terms of having the constraints all in one place. The interface was designed in hopes of catering to pipelines and schedulers beyond the examples in this repository and elsewhere at this time of writing. Also the scheduler step is a method of this pipeline interface and this may seem redundant given that the raw scheduler is accessible via the interface but this is the only way to constrain the scheduler step output to an output type befitting of the algorithm at hand (DDPO).

For a more detailed look into the interface and the associated default implementation, go [here](https://github.com/lvwerra/trl/tree/main/trl/models/modeling_sd_base.py)

Note that the default implementation has a LoRA implementation path and a non-LoRA based implementation path. The LoRA flag enabled by default and this can be turned off by passing in the flag to do so. LORA based training is faster and the LORA associated model hyperparameters responsible for model convergence aren't as finicky as non-LORA based training.

Also in addition, there is the expectation of providing a reward function and a prompt function. The reward function is used to evaluate the generated images and the prompt function is used to generate the prompts that are used to generate the images.

## Getting started with `examples/scripts/ddpo.py`

The `ddpo.py` script is a working example of using the `DDPO` trainer to finetune a Stable Diffusion model. This example explicitly configures a small subset of the overall parameters associated with the config object (`DDPOConfig`).

**Note:** one A100 GPU is recommended to get this running. Anything below a A100 will not be able to run this example script and even if it does via relatively smaller sized parameters, the results will most likely be poor.

Almost every configuration parameter has a default. There is only one commandline flag argument that is required of the user to get things up and running. The user is expected to have a [huggingface user access token](https://huggingface.co/docs/hub/security-tokens) that will be used to upload the model post finetuning to HuggingFace hub. The following bash command is to be entered to get things running

```batch
python ddpo.py --hf_user_access_token <token>
```

To obtain the documentation of `stable_diffusion_tuning.py`, please run `python stable_diffusion_tuning.py --help`

The following are things to keep in mind (The code checks this for you as well) in general while configuring the trainer (beyond the use case of using the example script)

- The configurable sample batch size (`--ddpo_config.sample_batch_size=6`) should be greater than or equal to the configurable training batch size (`--ddpo_config.train_batch_size=3`)
- The configurable sample batch size (`--ddpo_config.sample_batch_size=6`) must be divisible by the configurable train batch size (`--ddpo_config.train_batch_size=3`)
- The configurable sample batch size (`--ddpo_config.sample_batch_size=6`) must be divisible by both the configurable gradient accumulation steps (`--ddpo_config.train_gradient_accumulation_steps=1`) and the configurable accelerator processes count 

## Setting up the image logging hook function

Expect the function to be given a list of lists of the form
```python
[[image, prompt, prompt_metadata, rewards, reward_metadata], ...]

```
and `image`, `prompt`, `prompt_metadata`, `rewards`, `reward_metadata` are batched.
The last list in the lists of lists represents the last sample batch. You are likely to want to log this one
While you are free to log however you want the use of `wandb` or `tensorboard` is recommended.

### Key terms

- `rewards` : The rewards/score is a numerical associated with the generated image and is key to steering the RL process
- `reward_metadata` : The reward metadata is the metadata associated with the reward. Think of this as extra information payload delivered alongside the reward
- `prompt` : The prompt is the text that is used to generate the image
- `prompt_metadata` : The prompt metadata is the metadata associated with the prompt. A situation where this will not be empty is when the reward model comprises of a [`FLAVA`](https://huggingface.co/docs/transformers/model_doc/flava) setup where questions and ground answers (linked to the generated image) are expected with the generated image (See here: https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/rewards.py#L45)
- `image` : The image generated by the Stable Diffusion model

Example code for logging sampled images with `wandb` is given below.

```python
# for logging these images to wandb

def image_outputs_hook(image_data, global_step, accelerate_logger):
    # For the sake of this example, we only care about the last batch
    # hence we extract the last element of the list
    result = {}
    images, prompts, _, rewards, _ = image_data[-1]
    for i, image in enumerate(images):
        pil = Image.fromarray(
            (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        )
        pil = pil.resize((256, 256))
        result[f"{prompts[i]:.25} | {rewards[i]:.2f}"] = [pil]
    accelerate_logger.log_images(
        result,
        step=global_step,
    )

```

### Using the finetuned model

Assuming you've done with all the epochs and have pushed up your model to the hub, you can use the finetuned model as follows

```python

import torch
from trl import DefaultDDPOStableDiffusionPipeline

pipeline = DefaultDDPOStableDiffusionPipeline("metric-space/ddpo-finetuned-sd-model")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# memory optimization
pipeline.vae.to(device, torch.float16)
pipeline.text_encoder.to(device, torch.float16)
pipeline.unet.to(device, torch.float16)

prompts = ["squirrel", "crab", "starfish", "whale","sponge", "plankton"]
results = pipeline(prompts)

for prompt, image in zip(prompts,results.images):
    image.save(f"{prompt}.png")

```

## Credits

This work is heavily influenced by the repo [here](https://github.com/kvablack/ddpo-pytorch) and the associated paper [Training Diffusion Models
with Reinforcement Learning by Kevin Black, Michael Janner, Yilan Du, Ilya Kostrikov, Sergey Levine](https://huggingface.co/papers/2305.13301).

## DDPOTrainer

[[autodoc]] DDPOTrainer

## DDPOConfig

[[autodoc]] DDPOConfig

