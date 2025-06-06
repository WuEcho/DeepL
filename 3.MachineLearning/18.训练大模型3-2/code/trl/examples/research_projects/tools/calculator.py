# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

import numpy as np
import torch
from transformers import AutoTokenizer, load_tool

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, TextEnvironment


def generate_data(n):
    """Generate random arithmetic tasks and answers."""
    tasks, answers = [], []
    for _ in range(n):
        a = np.random.randint(0, 50)
        b = np.random.randint(0, 50)
        op = np.random.choice(["-", "+", "*"])
        tasks.append(f"\n\nWhat is {a} {op} {b}?")
        if op == "-":
            answers.append(a - b)
        elif op == "+":
            answers.append(a + b)
        else:
            answers.append(a * b)
    return tasks, answers


def exact_match_reward(responses, answers=None):
    """Reward if generated response contains correct answer."""
    rewards = []
    pattern = r"Result\s*=\s*(-?\d+(?:\.\d+)?)\s*<submit>"  # generated by chatGPT
    for response, answer in zip(responses, answers):
        reward = 0.0
        predicted_number = None
        match_pattern = re.findall(pattern, response)
        if match_pattern:
            predicted_number = float(match_pattern[0])
        if predicted_number is not None:
            if np.abs(predicted_number - answer) < 0.01:
                reward += 1.0
        rewards.append(torch.tensor(reward))
    return rewards


# set up models
model_id = "gpt2"
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# system prompt
prompt = """\
What is 13-3?

<request><SimpleCalculatorTool>13-3<call>10.0<response>

Result=10<submit>

What is 4*3?

<request><SimpleCalculatorTool>4*3<call>12.0<response>

Result=12<submit>"""

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": -1,
    "max_new_tokens": 32,
}

# trainer
ppo_config = PPOConfig(
    batch_size=256,
    learning_rate=1.41e-5,
    mini_batch_size=64,
    log_with="wandb",
)
ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer)

# text env
text_env = TextEnvironment(
    model,
    tokenizer,
    {"SimpleCalculatorTool": load_tool("ybelkada/simple-calculator")},
    exact_match_reward,
    prompt,
    generation_kwargs=generation_kwargs,
)

# main training loop
for _step in range(100):
    tasks, answers = generate_data(ppo_config.batch_size)
    queries, responses, masks, rewards, histories = text_env.run(tasks, answers=answers)
    train_stats = ppo_trainer.step(queries, responses, rewards, masks)

    response_texts = [tokenizer.decode(response) for response in responses]
    query_texts = [tokenizer.decode(query) for query in queries]
    texts = {"query": [qt.split("<submit>")[-1].strip() for qt in query_texts], "response": response_texts}
    ppo_trainer.log_stats(train_stats, texts, rewards, columns_to_log=["query", "response", "answer"])
ppo_trainer.save_pretrained(model_id + "-calculator")
