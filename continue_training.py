#!/usr/bin/env python3
"""
PRM800K Training Script - Resuming from checkpoint and optimized for A100
"""

import torch
import json
import random  # <-- Import random for proper sampling
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

print("🚀 Starting PRM800K Training on A100...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ============================================================================
# 1. DEFINE CHECKPOINT TO RESUME FROM
# ============================================================================
# This is the path to the checkpoint you want to resume
RESUME_CHECKPOINT_PATH = "./phi2-prm-a100/checkpoint-1000"
print(f"\n▶️ Resuming training from checkpoint: {RESUME_CHECKPOINT_PATH}")

# ============================================================================
# 2. LOAD FORMATTED DATA
# ============================================================================
print("\n📥 Loading formatted datasets...")

with open('train_formatted.json', 'r') as f:
    train_formatted = json.load(f)

with open('test_formatted.json', 'r') as f:
    test_formatted = json.load(f)

print(f"Total training examples: {len(train_formatted)}")
print(f"Total test examples: {len(test_formatted)}")

# ============================================================================
# 3. SAMPLE DATASET (10% for A100) - NOW WITH CORRECT SAMPLING
# ============================================================================
print("\n✂️ Sampling dataset for 3-hour training...")

# Use 10% of training data (~64K examples)
sample_size = len(train_formatted) // 10
# FIX: Use random.sample to get a proper random subset, not just the first 10%
print("...using random.sample for unbiased dataset.")
train_sample = random.sample(train_formatted, sample_size)

# Use 10% of test data
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
# 4. LOAD MODEL WITH QUANTIZATION
# ============================================================================
print("\n🤖 Loading Phi-2 model with 4-bit quantization...")

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✅ Model loaded!")

# ============================================================================
# 5. CONFIGURE LORA
# ============================================================================
print("\n⚙️ Configuring LoRA...")
print("   (Using r=16 to match checkpoint)")

# NOTE: This config MUST match the one used to create checkpoint-1000.
# We cannot change `r` or `target_modules` when resuming.
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.4f}")

# ============================================================================
# 6. TRAINING CONFIGURATION (Optimized for A100)
# ============================================================================
print("\n🔧 Setting up A100-optimized training configuration...")

training_args = TrainingArguments(
    output_dir="./phi2-prm-a100",  # Continue saving to the same directory
    
    # A100 can handle larger batches and more steps in 3 hours
    max_steps=4000,  # Trainer will resume and run up to this total step count
    
    # --- A100 OPTIMIZATIONS ---
    per_device_train_batch_size=32,  # INCREASED from 16
    per_device_eval_batch_size=32,   # INCREASED from 16
    gradient_accumulation_steps=1,   # DECREASED from 2
    # Effective batch size = 32 * 1 = 32 (Same as before, but runs faster)
    
    dataloader_num_workers=8,  # INCREASED from 4 (A100 can handle more)
    dataloader_pin_memory=True,
    # --- END OPTIMIZATIONS ---
    
    learning_rate=2e-4,
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
    
    # Performance optimizations
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
# 7. START TRAINING (RESUMING FROM CHECKPOINT)
# ============================================================================
print("\n" + "="*70)
print(f"🚀 RESUMING TRAINING FROM {RESUME_CHECKPOINT_PATH}")
print("="*70)
print(f"⏱️  Training up to {training_args.max_steps} total steps.")
print(f"📊 Training examples: {len(train_dataset)}")
print(f"⚡ Effective Batch Size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"💾 Checkpoints: ./phi2-prm-a100")
print("="*70 + "\n")

# This is the key change: pass the checkpoint path to trainer.train()
trainer.train(resume_from_checkpoint=RESUME_CHECKPOINT_PATH)

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)

# ============================================================================
# 8. SAVE FINAL MODEL
# ============================================================================
print("\n💾 Saving final model...")

# Save the fine-tuned model
model.save_pretrained("./phi2-prm-final")
tokenizer.save_pretrained("./phi2-prm-final")

print("✅ Model saved to ./phi2-prm-final")

# ============================================================================
# 9. SAVE TRAINING INFO
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
    "learning_rate": training_args.learning_rate,
}

with open("./phi2-prm-final/training_info.json", "w") as f:
    json.dump(training_info, f, indent=2)

print("\n✅ Training info saved!")

# ============================================================================
# 10. CREATE ARCHIVE FOR DOWNLOAD
# ============================================================================
print("\n📦 Creating archive for download...")

import os
# -f flag ensures it overwrites if the file already exists
os.system("tar -czf phi2-prm-final.tar.gz phi2-prm-final/")

print("✅ Archive created: phi2-prm-final.tar.gz")
print("\n" + "="*70)
print("🎉 ALL DONE!")
print("="*70)
print("\n📥 To download your model:")
print("   1. Download: phi2-prm-final.tar.gz")
print("   2. Extract: tar -xzf phi2-prm-final.tar.gz")
print("   3. Load with:")
print("      from peft import PeftModel, PeftConfig")
print("      from transformers import AutoModelForCausalLM")
print("      config = PeftConfig.from_pretrained('./phi2-prm-final')")
print("      model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)")
print("      model = PeftModel.from_pretrained(model, './phi2-prm-final')")
print("="*70 + "\n")