<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl_banner_dark.png">
</div>

# TRL - Transformer Reinforcement Learning

TRL is a full stack library where we provide a set of tools to train transformer language models with Reinforcement Learning, from the Supervised Fine-tuning step (SFT), Reward Modeling step (RM) to the Proximal Policy Optimization (PPO) step. 
The library is integrated with 🤗 [transformers](https://github.com/huggingface/transformers).

## Learn

Learn post-training with TRL and other libraries in 🤗 [smol course](https://github.com/huggingface/smol-course).

## API documentation

- [Model Classes](models): *A brief overview of what each public model class does.*
- [`SFTTrainer`](sft_trainer): *Supervise Fine-tune your model easily with `SFTTrainer`*
- [`RewardTrainer`](reward_trainer): *Train easily your reward model using `RewardTrainer`.*
- [`PPOTrainer`](ppo_trainer): *Further fine-tune the supervised fine-tuned model using PPO algorithm*
- [Best-of-N Sampling](best-of-n): *Use best of n sampling as an alternative way to sample predictions from your active model*
- [`DPOTrainer`](dpo_trainer): *Direct Preference Optimization training using `DPOTrainer`.*
- [`TextEnvironment`](text_environments): *Text environment to train your model using tools with RL.*

## Examples

- [Sentiment Tuning](sentiment_tuning): *Fine tune your model to generate positive movie contents*
- [Training with PEFT](lora_tuning_peft): *Memory efficient RLHF training using adapters with PEFT*
- [Detoxifying LLMs](detoxifying_a_lm): *Detoxify your language model through RLHF*
- [StackLlama](using_llama_models): *End-to-end RLHF training of a Llama model on Stack exchange dataset*
- [Learning with Tools](learning_tools): *Walkthrough of using `TextEnvironments`*
- [Multi-Adapter Training](multi_adapter_rl): *Use a single base model and multiple adapters for memory efficient end-to-end training*


## Blog posts

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/dpo_vlm">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/dpo_vlm/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">Published on July 10, 2024</p>
      <p class="text-gray-700">Preference Optimization for Vision Language Models with TRL</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/putting_rl_back_in_rlhf_with_rloo/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">Published on June 12, 2024</p>
      <p class="text-gray-700">Putting RL back in RLHF</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/trl-ddpo">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/166_trl_ddpo/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">Published on September 29, 2023</p>
      <p class="text-gray-700">Finetune Stable Diffusion Models with DDPO via TRL</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/dpo-trl">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/157_dpo_trl/dpo_thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">Published on August 8, 2023</p>
      <p class="text-gray-700">Fine-tune Llama 2 with DPO</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/stackllama">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/138_stackllama/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">Published on April 5, 2023</p>
      <p class="text-gray-700">StackLLaMA: A hands-on guide to train LLaMA with RLHF</p>
   </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/trl-peft">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/133_trl_peft/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">Published on March 9, 2023</p>
      <p class="text-gray-700">Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/rlhf">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/120_rlhf/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">Published on December 9, 2022</p>
      <p class="text-gray-700">Illustrating Reinforcement Learning from Human Feedback</p>
    </a>
  </div>
</div>
