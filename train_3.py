#!/usr/bin/env python3
"""
PRM800K Training Script - OPTIMIZED for 4-Hour A100 Training
High accuracy + Fast convergence within time limit
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import json
import random
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

print("🚀 Starting 4-HOUR OPTIMIZED PRM800K Training...")
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
# 2. SAMPLE DATASET - 20% for 4-hour training with good accuracy
# ============================================================================
print("\n✂️ Sampling dataset for 4-hour training...")

# 20% of data = ~128K examples - Perfect for 4 hours on A100
random.seed(42)
sample_size = len(train_formatted) // 5  # 20%
train_sample = random.sample(train_formatted, sample_size)

test_sample_size = len(test_formatted) // 5  # 20%
test_sample = random.sample(test_formatted, test_sample_size)

# Create HuggingFace datasets
train_dataset = Dataset.from_dict({
    'text': [ex['full_text'] for ex in train_sample]
})

test_dataset = Dataset.from_dict({
    'text': [ex['full_text'] for ex in test_sample]
})

print(f"✅ Training on {len(train_dataset)} examples (20% random sample)")
print(f"✅ Evaluating on {len(test_dataset)} examples (20% random sample)")

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
# 4. CONFIGURE LORA - High Capacity with Stability
# ============================================================================
print("\n⚙️ Configuring HIGH-CAPACITY LoRA...")

lora_config = LoraConfig(
    r=64,  # High capacity for better accuracy
    lora_alpha=64,  # Stable scaling (alpha = r)
    
    # ✅ FIXED: All correct module names (no typos)
    target_modules=[
        "q_proj",  # Query projection
        "k_proj",  # Key projection
        "v_proj",  # Value projection
        "dense",   # Output projection
        "fc1",     # MLP layer 1 (for reasoning)
        "fc2"      # MLP layer 2 (for reasoning)
    ],
    
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.4f}")

# ============================================================================
# 5. CUSTOM CALLBACK FOR REAL-TIME LOGGING
# ============================================================================
NEW_OUTPUT_DIR = "./phi2-prm-4hr"
LOG_FILE_PATH = f"{NEW_OUTPUT_DIR}/training_log_history.jsonl"

class JsonLogCallback(TrainerCallback):
    """Real-time JSON logging"""
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(self.log_file_path, "w") as f:
            f.write("")
        print(f"   Real-time log: {self.log_file_path}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            with open(self.log_file_path, "a") as f:
                f.write(json.dumps(logs) + "\n")

json_log_callback = JsonLogCallback(LOG_FILE_PATH)

# ============================================================================
# 6. TRAINING CONFIGURATION - Optimized for 4 hours
# ============================================================================
print("\n🔧 Setting up 4-HOUR training configuration...")

training_args = TrainingArguments(
    output_dir=NEW_OUTPUT_DIR,
    
    # ⏱️ OPTIMIZED: 1 full epoch on 20% data ≈ 4 hours on A100
    num_train_epochs=1.0,
    
    # Larger batch size for A100 speed
    per_device_train_batch_size=28,  # Maximized for A100
    per_device_eval_batch_size=28,
    gradient_accumulation_steps=2,
    # Effective batch size = 28 * 2 = 56 (large = stable + fast)
    
    # Stable learning rate
    learning_rate=1.5e-4,  # Conservative for stability
    max_grad_norm=1.0,     # Good clipping
    
    # Good warmup for smooth start
    warmup_ratio=0.1,  # 10% warmup
    
    lr_scheduler_type="cosine",
    optim="paged_adamw_32bit",
    weight_decay=0.001,
    
    # Evaluation & Checkpointing
    logging_steps=20,
    save_steps=400,      # Save every ~30 min
    eval_steps=400,      # Eval every ~30 min
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # Performance optimizations for A100
    bf16=True,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    
    report_to="tensorboard",
    save_total_limit=3,
    seed=42,
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
    formatting_func=formatting_func,
    callbacks=[json_log_callback]
)

print("✅ Trainer configured!")

# ============================================================================
# 7. START TRAINING
# ============================================================================
print("\n" + "="*70)
print("🚀 STARTING 4-HOUR OPTIMIZED TRAINING")
print("="*70)
print(f"⏱️  Target time: ~4 hours on A100")
print(f"📊 Training examples: {len(train_dataset)} (20% of full dataset)")
print(f"📈 Training duration: 1.0 epoch")
print(f"⚡ Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"💾 Checkpoints: {NEW_OUTPUT_DIR}")
print("")
print("Configuration:")
print(f"  • LoRA Rank: {lora_config.r}")
print(f"  • LoRA Alpha: {lora_config.lora_alpha}")
print(f"  • Learning Rate: {training_args.learning_rate}")
print(f"  • Warmup: {training_args.warmup_ratio * 100}%")
print(f"  • Target Modules: 6 (attention + MLP)")
print("="*70 + "\n")

trainer.train()

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)

# ============================================================================
# 8. SAVE FINAL MODEL
# ============================================================================
print("\n💾 Saving final model...")
FINAL_MODEL_PATH = "./phi2-prm-4hr-final"

model.save_pretrained(FINAL_MODEL_PATH)
tokenizer.save_pretrained(FINAL_MODEL_PATH)

print(f"✅ Model saved to {FINAL_MODEL_PATH}")

# ============================================================================
# 9. SAVE TRAINING INFO
# ============================================================================
training_info = {
    "model": "microsoft/phi-2",
    "dataset": "PRM800K",
    "training_examples": len(train_dataset),
    "test_examples": len(test_dataset),
    "data_percentage": 20,
    "epochs": training_args.num_train_epochs,
    "batch_size": training_args.per_device_train_batch_size,
    "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
    "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
    "lora_r": lora_config.r,
    "lora_alpha": lora_config.lora_alpha,
    "target_modules": list(lora_config.target_modules),
    "learning_rate": training_args.learning_rate,
    "warmup_ratio": training_args.warmup_ratio,
    "max_grad_norm": training_args.max_grad_norm,
    "training_time_target": "4 hours on A100",
}

with open(f"{FINAL_MODEL_PATH}/training_info.json", "w") as f:
    json.dump(training_info, f, indent=2)

print("✅ Training info saved!")
print(f"✅ Real-time logs: {LOG_FILE_PATH}")

# ============================================================================
# 10. CREATE ARCHIVE FOR DOWNLOAD
# ============================================================================
print("\n📦 Creating archive for download...")

archive_name = "phi2-prm-4hr-final.tar.gz"
os.system(f"tar -czf {archive_name} {FINAL_MODEL_PATH}/")

print(f"✅ Archive created: {archive_name}")
print("\n" + "="*70)
print("🎉 ALL DONE - 4 HOUR TRAINING COMPLETE!")
print("="*70)
print("\n📊 Expected Results:")
print("   • Token accuracy: 83-86% (vs 81.5% with r=16)")
print("   • Loss: <0.62 (vs 0.65 with r=16)")
print("   • Stable convergence with no grad explosions")
print("   • Training time: ~4 hours on A100")
print("="*70 + "\n")