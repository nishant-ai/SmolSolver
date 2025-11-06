#!/usr/bin/env python3
"""
PRM800K Training Script - NEW High-Capacity Run
(Optimized for better convergence, not just speed)
"""

import torch
import json
import random
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

print("🚀 Starting NEW High-Capacity PRM800K Training...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ============================================================================
# 1. LOAD FORMATTED DATA
# ============================================================================
print("\n📥 Loading formatted datasets...")

with open('train_formatted.json', 'r') as f:
    train_formatted = json.load(f)

with open('test_formatted.json', 'r') as f:
    test_formatted = json.load(f)

print(f"Total training examples: {len(train_formatted)}")
print(f"Total test examples: {len(test_formatted)}")

# ============================================================================
# 2. SAMPLE DATASET (Corrected Sampling)
# ============================================================================
print("\n✂️ Sampling dataset...")

# We are using 10% of the data. If this run goes well,
# the next step would be to increase this to 20% or 30%.
sample_size = len(train_formatted) // 10
train_sample = random.sample(train_formatted, sample_size)

test_sample_size = len(test_formatted) // 10
test_sample = random.sample(test_formatted, test_sample_size)

# Create HuggingFace datasets
train_dataset = Dataset.from_dict({
    'text': [ex['full_text'] for ex in train_sample]
})

test_dataset = Dataset.from_dict({
    'text': [ex['full_text'] for ex in test_sample]
})

print(f"✅ Training on {len(train_dataset)} examples (10% random sample)")
print(f"✅ Evaluating on {len(test_dataset)} examples (10% random sample)")

# ============================================================================
# 3. LOAD MODEL WITH QUANTIZATION
# ============================================================================
print("\n🤖 Loading Phi-2 model with 4-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✅ Model loaded!")

# ============================================================================
# 4. CONFIGURE LORA (*** THE KEY CHANGE IS HERE ***)
# ============================================================================
print("\n⚙️ Configuring HIGH-CAPACITY LoRA...")

lora_config = LoraConfig(
    r=64,  # <-- INCREASED from 16 to 64. This is the "rank" or "capacity".
    lora_alpha=128, # <-- INCREASED from 64. Set to 2*r, a common heuristic for stronger learning.
    
    # *** NEW ***
    # Target the "brain" (MLP layers) in addition to attention layers
    # This is critical for reasoning tasks.
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "dense",
        "fc1",
        "fc2"
    ],
    
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.4f}")
print("   (This trainable % will be higher than before, this is the goal!)")

# ============================================================================
# 5. TRAINING CONFIGURATION (Optimized for A100 Speed)
# ============================================================================
print("\n🔧 Setting up A100-optimized training configuration...")

# We'll create a new output directory for this new run
NEW_OUTPUT_DIR = "./phi2-prm-r64"
print(f"   Saving checkpoints to: {NEW_OUTPUT_DIR}")

training_args = TrainingArguments(
    output_dir=NEW_OUTPUT_DIR,
    max_steps=4000,
    
    # --- A100 Speed Optimizations ---
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    # --- End Speed Optimizations ---
    
    learning_rate=2.5e-4, # <-- SLIGHTLY INCREASED from 2e-4 for faster convergence
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="paged_adamw_32bit",
    weight_decay=0.001,
    
    # Logging & checkpointing
    logging_steps=25,
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    
    bf16=True,
    gradient_checkpointing=True,
    
    report_to="tensorboard",
    save_total_limit=2,
)

# Formatting function
def formatting_func(example):
    return example['text']

# Create trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    formatting_func=formatting_func
)

print("✅ Trainer configured!")

# ============================================================================
# 6. START TRAINING
# ============================================================================
print("\n" + "="*70)
print("🚀 STARTING NEW HIGH-CAPACITY TRAINING (FROM SCRATCH)")
print("="*70)
print("🚀 STARTING NEW HIGH-CAPACITY TRAINING (FROM SCRATCH)")
print("="*70)
print(f"⏱️  Training up to {training_args.max_steps} total steps.")
print(f"📊 Training examples: {len(train_dataset)}")
print(f"⚡ Effective Batch Size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"💾 Checkpoints: {NEW_OUTPUT_DIR}")
print(f"💡 LoRA Rank (r): {lora_config.r}")
print(f"💡 LoRA Alpha: {lora_config.lora_alpha}")
print(f"💡 Learning Rate: {training_args.learning_rate}")
print("="*70 + "\n")

# Note: We are NOT passing `resume_from_checkpoint`
trainer.train()

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)

# ============================================================================
# 7. SAVE FINAL MODEL
# ============================================================================
print("\n💾 Saving final model...")
FINAL_MODEL_PATH = "./phi2-prm-r64-final"

model.save_pretrained(FINAL_MODEL_PATH)
tokenizer.save_pretrained(FINAL_MODEL_PATH)

print(f"✅ Model saved to {FINAL_MODEL_PATH}")

# =================================S===========================================
# 8. SAVE TRAINING INFO
# ============================================================================
training_info = {
    "model": "microsoft/phi-2",
    "dataset": "PRM800K",
    "training_examples": len(train_dataset),
    "test_examples": len(test_dataset),
    "total_steps": training_args.max_steps,
    "batch_size": training_args.per_device_train_batch_size,
    "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
    "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
    "lora_r": lora_config.r,
    "lora_alpha": lora_config.lora_alpha,
    "target_modules": list(lora_config.target_modules),
    "learning_rate": training_args.learning_rate,
}

with open(f"{FINAL_MODEL_PATH}/training_info.json", "w") as f:
    json.dump(training_info, f, indent=2)

print("\n✅ Training info saved!")

# ============================================================================
# 9. CREATE ARCHIVE FOR DOWNLOAD
# ============================================================================
print("\n📦 Creating archive for download...")

import os
archive_name = "phi2-prm-r64-final.tar.gz"
os.system(f"tar -czf {archive_name} {FINAL_MODEL_PATH}/")

print(f"✅ Archive created: {archive_name}")
print("\n" + "="*70)
print("🎉 ALL DONE!")
print("="*70)