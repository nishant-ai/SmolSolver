#!/usr/bin/env python3
"""
PRM800K Training Script - RESUME from Checkpoint
Continues training from where you left off
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
import glob

print("🔄 RESUMING PRM800K Training from Checkpoint...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ============================================================================
# 0. FIND LAST CHECKPOINT
# ============================================================================
CHECKPOINT_DIR = "./phi2-prm-4hr"

# Find all checkpoints
checkpoints = glob.glob(f"{CHECKPOINT_DIR}/checkpoint-*")
if not checkpoints:
    print("❌ No checkpoints found! Run the training script first.")
    exit(1)

# Sort by step number and get the last one
checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
last_checkpoint = checkpoints[-1]
checkpoint_step = int(last_checkpoint.split('-')[-1])

print(f"\n📂 Found checkpoint: {last_checkpoint}")
print(f"   Resuming from step: {checkpoint_step}")

# ============================================================================
# 1. LOAD FORMATTED DATA (Same as before)
# ============================================================================
print("\n📥 Loading formatted datasets...")

with open('train_formatted.json', 'r') as f:
    train_formatted = json.load(f)

with open('test_formatted.json', 'r') as f:
    test_formatted = json.load(f)

print(f"Total training examples: {len(train_formatted)}")
print(f"Total test examples: {len(test_formatted)}")

# ============================================================================
# 2. SAMPLE DATASET (MUST USE SAME SEED!)
# ============================================================================
print("\n✂️ Sampling dataset (using same seed as original)...")

# ⚠️ IMPORTANT: Use same seed and same sampling to get same data!
random.seed(42)  # Same seed as original training
sample_size = len(train_formatted) // 5  # 20%
train_sample = random.sample(train_formatted, sample_size)

test_sample_size = len(test_formatted) // 5  # 20%
test_sample = random.sample(test_formatted, test_sample_size)

train_dataset = Dataset.from_dict({
    'text': [ex['full_text'] for ex in train_sample]
})

test_dataset = Dataset.from_dict({
    'text': [ex['full_text'] for ex in test_sample]
})

print(f"✅ Training on {len(train_dataset)} examples")
print(f"✅ Evaluating on {len(test_dataset)} examples")

# ============================================================================
# 3. LOAD MODEL FROM CHECKPOINT
# ============================================================================
print("\n🤖 Loading model from checkpoint...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ⚠️ IMPORTANT: Load LoRA weights from checkpoint
print(f"   Loading LoRA weights from {last_checkpoint}...")
model = PeftModel.from_pretrained(model, last_checkpoint, is_trainable=True)

print("✅ Model loaded from checkpoint!")

# ============================================================================
# 4. CUSTOM CALLBACK FOR LOGGING (Continue appending)
# ============================================================================
LOG_FILE_PATH = f"{CHECKPOINT_DIR}/training_log_history.jsonl"

class JsonLogCallback(TrainerCallback):
    """Real-time JSON logging (appends to existing file)"""
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        print(f"   Continuing log: {self.log_file_path}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            with open(self.log_file_path, "a") as f:
                f.write(json.dumps(logs) + "\n")

json_log_callback = JsonLogCallback(LOG_FILE_PATH)

# ============================================================================
# 5. TRAINING CONFIGURATION (SAME AS ORIGINAL)
# ============================================================================
print("\n🔧 Setting up training configuration...")

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,  # Same output directory
    
    # Same configuration as original
    num_train_epochs=1.0,  # Will continue from where it stopped
    
    per_device_train_batch_size=28,
    per_device_eval_batch_size=28,
    gradient_accumulation_steps=2,
    
    learning_rate=1.5e-4,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    
    lr_scheduler_type="cosine",
    optim="paged_adamw_32bit",
    weight_decay=0.001,
    
    logging_steps=20,
    save_steps=400,
    eval_steps=400,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
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

# Compute metrics
import numpy as np

def compute_metrics(eval_preds):
    """Compute token-level accuracy"""
    predictions, labels = eval_preds
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    predictions = np.argmax(predictions, axis=-1)
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)
    
    mask = labels != -100
    
    if mask.sum() == 0:
        return {"token_accuracy": 0.0}
    
    correct = (predictions[mask] == labels[mask]).sum()
    total = mask.sum()
    accuracy = correct / total
    
    return {
        "token_accuracy": float(accuracy),
        "num_tokens": int(total),
    }

# Create trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    formatting_func=formatting_func,
    callbacks=[json_log_callback],
    compute_metrics=compute_metrics,
)

print("✅ Trainer configured!")

# ============================================================================
# 6. RESUME TRAINING
# ============================================================================
print("\n" + "="*70)
print("🔄 RESUMING TRAINING FROM CHECKPOINT")
print("="*70)
print(f"📂 Checkpoint: {last_checkpoint}")
print(f"📊 Resuming from step: {checkpoint_step}")
print(f"⏱️  Continuing training...")
print("="*70 + "\n")

# ⚠️ KEY LINE: Pass the checkpoint path
trainer.train(resume_from_checkpoint=last_checkpoint)

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)

# ============================================================================
# 7. SAVE FINAL MODEL
# ============================================================================
print("\n💾 Saving final model...")
FINAL_MODEL_PATH = "./phi2-prm-4hr-final"

model.save_pretrained(FINAL_MODEL_PATH)
tokenizer.save_pretrained(FINAL_MODEL_PATH)

print(f"✅ Model saved to {FINAL_MODEL_PATH}")

# ============================================================================
# 8. SAVE TRAINING INFO
# ============================================================================
training_info = {
    "model": "microsoft/phi-2",
    "dataset": "PRM800K",
    "training_examples": len(train_dataset),
    "test_examples": len(test_dataset),
    "data_percentage": 20,
    "resumed_from": last_checkpoint,
    "resumed_from_step": checkpoint_step,
    "total_epochs": training_args.num_train_epochs,
}

with open(f"{FINAL_MODEL_PATH}/training_info.json", "w") as f:
    json.dump(training_info, f, indent=2)

print("✅ Training info saved!")
print(f"✅ Full logs at: {LOG_FILE_PATH}")

# ============================================================================
# 9. CREATE ARCHIVE
# ============================================================================
print("\n📦 Creating archive...")

archive_name = "phi2-prm-4hr-final-resumed.tar.gz"
os.system(f"tar -czf {archive_name} {FINAL_MODEL_PATH}/")

print(f"✅ Archive created: {archive_name}")
print("\n" + "="*70)
print("🎉 RESUMED TRAINING COMPLETE!")
print("="*70 + "\n")