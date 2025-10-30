Project Plan: Baseline Generator-Verifier System for Math Reasoning

Objective: To build a high-performing math reasoning system by creating two specialized Small Language Models (SLMs): a Generator that solves problems and a Verifier that provides human-like feedback on each reasoning step.

Core Components:

Base Model: microsoft/Phi-3-mini-4k-instruct (for both Generator and Verifier)

Generator Dataset: math-qa/GSM8K (A standard dataset of math problems and solutions)

Verifier Dataset: openai/prm800k (A dataset of human-labeled reasoning steps with feedback)

## Phase 1: Environment Setup

Goal: Prepare your development environment with all necessary tools and libraries.

Install Python & PyTorch: Ensure you have a recent version of Python (3.9+) and PyTorch installed, preferably with CUDA support for GPU acceleration.

Install Hugging Face Libraries: These are the core tools for your project.

pip install transformers datasets accelerate bitsandbytes trl


transformers: For loading models and tokenizers.

datasets: For loading and processing GSM8K and prm800k.

accelerate: For efficient multi-GPU/mixed-precision training.

bitsandbytes: For loading models in 4-bit (QLoRA) to save memory.

trl: For the SFTTrainer, an easy-to-use trainer for fine-tuning.

Hugging Face Login: Log in to your Hugging Face account to access gated models like Phi-3.

huggingface-cli login


## Phase 2: Building the Generator Model

Goal: Fine-tune a Phi-3-mini model to be a proficient math problem solver that generates step-by-step solutions (chains of thought).

Load Dataset: Load the GSM8K dataset (main-train split).

from datasets import load_dataset
gsm8k_dataset = load_dataset("math-qa/GSM8K", "main", split="train")


Format the Data: Create a "prompt-response" format. Phi-3-mini-instruct uses a specific chat template (<|user|>...<|end|><|assistant|>...<|end|>). You'll format your data to follow this.

Prompt (User): "Solve the following math problem: [Problem text from GSM8K]"

Response (Assistant): "[Solution text from GSM8K]"

Load Model & Tokenizer: Load the Phi-3-mini model and its tokenizer. Use 4-bit quantization (QLoRA) to make it fit on a single consumer GPU.

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_id = "microsoft/Phi-3-mini-4k-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

generator_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
generator_tokenizer = AutoTokenizer.from_pretrained(model_id)


Fine-Tuning: Use the SFTTrainer from the trl library to fine-tune the model. This trainer is ideal because it handles the data formatting and training loop for you.

Save Model: After training, save your fine-tuned Generator. This is now your specialized problem-solver.

generator_model.save_pretrained("./my-math-generator")
generator_tokenizer.save_pretrained("./my-math-generator")


## Phase 3: Building the Verifier Model (The PRM)

Goal: Fine-tune a second, fresh Phi-3-mini model to act as a Process Reward Model (PRM). It will learn to judge a reasoning step and provide human-like feedback.

Load Dataset: Load the prm800k dataset.

prm_dataset = load_dataset("openai/prm800k", split="train")


Format the Data: This is the most crucial step. You will filter the dataset and format it to teach the model to output human-like feedback.

Iterate through the dataset. Each entry contains a prompt (the problem), a response (the step-by-step solution), and a human_labels list.

You are interested in the human_labels, which contain the rating ("correct" or "incorrect") and the comment.

Prompt (User):

You are a math expert. Evaluate the following reasoning step.

Problem:
[Problem text]

Reasoning Step:
[Text of the specific step]

Is this step correct or incorrect? Provide your reasoning.


Response (Assistant):

[Rating, from label]. [Comment, from label]

Example: "Incorrect. The step incorrectly uses addition instead of multiplication to calculate the total cost."

Example: "Correct. This step correctly applies the distributive property."


You will need to write a script to transform the prm800k structure into this text-based format.

Load Model & Tokenizer: Load a fresh, pre-trained instance of Phi-3-mini (just as in Phase 2). It's critical that this model does not have the Generator's fine-tuning.

Fine-Tuning: Fine-tune this second model on your newly formatted prm800k dataset. It will learn to become a verifier that outputs a judgment and an explanation.

Save Model: Save your fine-tuned Verifier.

verifier_model.save_pretrained("./my-math-verifier")
verifier_tokenizer.save_pretrained("./my-math-verifier")


## Phase 4: Integration and Inference Loop

Goal: Make the Generator and Verifier work together in a "self-correction" loop.

Load Both Models: Load your fine-tuned my-math-generator and my-math-verifier.

Create the Loop: Write a Python script to manage the reasoning process.

Start: Give the Generator the initial math problem.

Generate Step: Instruct the Generator to produce only the next reasoning step.

Verify Step: Take the step from the Generator and feed it to the Verifier using its specific prompt format.

Get Feedback: Capture the Verifier's response (e.g., "Incorrect. You forgot to carry the one.").

Make Decision:

If the Verifier's response starts with "Correct", append the step to your final solution and ask the Generator for the next step.

If the Verifier's response starts with "Incorrect", append the Generator's bad step and the Verifier's feedback to the conversation history. Then, ask the Generator to try again, now informed of its mistake.

End: The loop finishes when the Generator produces a final answer (e.g., "The final answer is...").

## Phase 5: Evaluation

Goal: Test how well your complete system performs.

Test Dataset: Use the "test" split of the GSM8K dataset, which neither of your models has ever seen.

Run System: Run your complete Generator-Verifier system (from Phase 4) on each problem in the test set.

Metric: The primary metric is accuracy. Compare the final answer produced by your system against the ground-truth answer in the test set.

Analyze Results:

What is the final pass@1 accuracy of your system?

How does this compare to the baseline (running the Generator without the Verifier)?

Read the conversation history. How many times did the Verifier correctly catch an error? How many times did the Generator successfully self-correct based on the feedback?