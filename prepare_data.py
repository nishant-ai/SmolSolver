#!/usr/bin/env python3
"""
PRM800K Data Preprocessing Script
Prepares train_formatted.json and test_formatted.json
"""

import json
from pathlib import Path
from huggingface_hub import hf_hub_download
from datasets import Dataset, DatasetDict

def load_prm800k_direct(split="all", cache_dir=None, limit=None):
    """
    Load PRM800K dataset directly from JSONL files
    
    Args:
        split: "train", "test", or "all" (default: "all")
        cache_dir: Directory to cache downloaded files
        limit: Limit number of examples per split (for testing)
    
    Returns:
        Dataset or DatasetDict depending on split
    """
    repo_id = "peiyi9979/math-shepherd-prm800k"
    
    # Define files for each split
    files = {
        "train": ["phase1_train.jsonl", "phase2_train.jsonl"],
        "test": ["phase1_test.jsonl", "phase2_test.jsonl"]
    }
    
    def load_split(split_name):
        all_data = []
        for filename in files[split_name]:
            print(f"   📄 Downloading {filename}...")
            filepath = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                cache_dir=cache_dir
            )
            
            print(f"   📖 Loading {filename}...")
            with open(filepath, 'r') as f:
                data = [json.loads(line) for line in f]
                if limit:
                    data = data[:limit]
                all_data.extend(data)
                print(f"      ✅ Loaded {len(data)} examples")
        
        return all_data
    
    if split == "all":
        print("📥 Loading PRM800K (test split)...")
        test_data = load_split("test")
        print("\n✅ Dataset loaded!")
        
        print("📥 Loading PRM800K (train split)...")
        train_data = load_split("train")
        print("\n✅ Dataset loaded!")
        
        return DatasetDict({
            "train": Dataset.from_list(train_data),
            "test": Dataset.from_list(test_data)
        })
    else:
        print(f"📥 Loading PRM800K ({split} split)...")
        data = load_split(split)
        print("\n✅ Dataset loaded!")
        return Dataset.from_list(data)


def prepare_dataset(data, output_file='formatted.json', include_negatives=True, max_samples=None):
    """
    Prepare PRM800K dataset for training
    """
    print("Processing PRM800K data...")
    
    formatted_examples = []
    
    for idx, item in enumerate(data):
        if max_samples and idx >= max_samples:
            break
            
        question = item['question']
        
        # Process each step with its label
        for step_data in item['label']:
            step_text = step_data['text']
            rating = step_data['label']
            
            # Skip negative examples if not including them
            if not include_negatives and rating == -1:
                continue
            
            # Format as: Question -> Step -> Rating
            full_text = f"""Problem: {question}

Step: {step_text}

Rating: {rating}"""
            
            formatted_examples.append({
                'full_text': full_text,
                'question': question,
                'step': step_text,
                'rating': rating
            })
    
    # Save to JSON
    print(f"\nTotal examples extracted: {len(formatted_examples)}")
    with open(output_file, 'w') as f:
        json.dump(formatted_examples, f)
    print(f"Saved to {output_file}")
    
    return formatted_examples


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("PRM800K DATA PREPROCESSING")
    print("="*70)
    
    # Load dataset
    print("\n📥 Loading PRM800K dataset...")
    dataset = load_prm800k_direct(split="all")
    
    train_data = dataset['train']
    test_data = dataset['test']
    
    print(f"\n✅ Loaded {len(train_data)} training examples")
    print(f"✅ Loaded {len(test_data)} test examples")
    
    # Prepare training data
    print("\n" + "="*70)
    print("PREPARING TRAINING DATA")
    print("="*70)
    train_formatted = prepare_dataset(
        train_data,
        output_file='train_formatted.json',
        include_negatives=True,
        max_samples=None  # Use all data
    )
    
    # Prepare test data
    print("\n" + "="*70)
    print("PREPARING TEST DATA")
    print("="*70)
    test_formatted = prepare_dataset(
        test_data,
        output_file='test_formatted.json',
        include_negatives=True,
        max_samples=None
    )
    
    # Print statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Training examples: {len(train_formatted)}")
    print(f"Test examples: {len(test_formatted)}")
    
    # Rating distribution
    from collections import Counter
    train_ratings = Counter([ex['rating'] for ex in train_formatted])
    print(f"\nRating distribution (train):")
    for rating in sorted(train_ratings.keys()):
        count = train_ratings[rating]
        pct = 100 * count / len(train_formatted)
        print(f"  Rating {rating}: {count} ({pct:.1f}%)")
    
    # Sample example
    print("\n" + "="*70)
    print("SAMPLE EXAMPLE")
    print("="*70)
    print(train_formatted[0]['full_text'][:500])
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  📄 train_formatted.json")
    print("  📄 test_formatted.json")
    print("\nNow you can run:")
    print("  python train_prm800k_a100.py")
    print("="*70 + "\n")