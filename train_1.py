#!/usr/bin/env python3
"""
PRM800K Training Script - Optimized for 1x A100 (3 hours)
"""

import torch
import json
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
# 2. SAMPLE DATASET (10% for A100 - more than 2xT4)
# ============================================================================
print("\n✂️ Sampling dataset for 3-hour training...")

# Use 10% of training data (~64K examples - A100 can handle more)
sample_size = len(train_formatted) // 10
train_sample = [train_formatted[i] for i in range(sample_size)]

# Use 10% of test data
test_sample_size = len(test_formatted) // 10
test_sample = [test_formatted[i] for i in range(test_sample_size)]

# Create HuggingFace datasets
train_dataset = Dataset.from_dict({
    'text': [ex['full_text'] for ex in train_sample]
})

test_dataset = Dataset.from_dict({
    'text': [ex['full_text'] for ex in test_sample]
})

print(f"✅ Training on {len(train_dataset)} examples (10% sample)")
print(f"✅ Evaluating on {len(test_dataset)} examples (10% sample)")

# ============================================================================
# 3. LOAD MODEL WITH QUANTIZATION
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
# 4. CONFIGURE LORA
# ============================================================================
print("\n⚙️ Configuring LoRA...")

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
# 5. TRAINING CONFIGURATION (Optimized for A100 - 3 hours)
# ============================================================================
print("\n🔧 Setting up training configuration...")

training_args = TrainingArguments(
    output_dir="./phi2-prm-a100",
    
    # A100 can handle larger batches and more steps in 3 hours
    max_steps=4000,  # ~3 hours on A100
    per_device_train_batch_size=16,  # A100 has 40-80GB VRAM
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    # Effective batch size = 16 * 2 = 32
    
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
    dataloader_num_workers=4,  # A100 can handle more workers
    dataloader_pin_memory=True,
    
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
print("🚀 STARTING TRAINING")
print("="*70)
print(f"⏱️  Expected time: ~3 hours on A100")
print(f"📊 Training examples: {len(train_dataset)}")
print(f"📈 Total steps: 4000")
print(f"💾 Checkpoints: ./phi2-prm-a100")
print("="*70 + "\n")

trainer.train()

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)

# ============================================================================
# 7. SAVE FINAL MODEL
# ============================================================================
print("\n💾 Saving final model...")

# Save the fine-tuned model
model.save_pretrained("./phi2-prm-final")
tokenizer.save_pretrained("./phi2-prm-final")

print("✅ Model saved to ./phi2-prm-final")

# ============================================================================
# 8. SAVE TRAINING INFO
# ============================================================================
training_info = {
    "model": "microsoft/phi-2",
    "dataset": "PRM800K",
    "training_examples": len(train_dataset),
    "test_examples": len(test_dataset),
    "total_steps": 4000,
    "batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "lora_r": 16,
    "lora_alpha": 16,
    "learning_rate": 2e-4,
}

with open("./phi2-prm-final/training_info.json", "w") as f:
    json.dump(training_info, f, indent=2)

print("\n✅ Training info saved!")

# ============================================================================
# 9. CREATE ARCHIVE FOR DOWNLOAD
# ============================================================================
print("\n📦 Creating archive for download...")

import os
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
print("      config = PeftConfig.from_pretrained('./phi2-prm-final')")
print("      model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)")
print("      model = PeftModel.from_pretrained(model, './phi2-prm-final')")
print("="*70 + "\n")
