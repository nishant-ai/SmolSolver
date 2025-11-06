#!/usr/bin/env python3
"""
PRM800K Data Preprocessing Script
Based on Verifier.ipynb - Creates train_formatted.json and test_formatted.json
"""

import json
import random
from pathlib import Path
from typing import Dict, List
from huggingface_hub import hf_hub_download
from datasets import Dataset, DatasetDict
from tqdm import tqdm

# Set seed for reproducibility
random.seed(42)

def load_prm800k_direct(split="all", cache_dir=None, limit=None):
    """
    Load PRM800K dataset directly from JSONL files

    Args:
        split: "train", "test", or "all" (default: "all")
        cache_dir: Directory to cache downloaded files
        limit: Limit number of examples per split (for testing)

    Returns:
        Dataset or DatasetDict depending on split parameter
    """

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "prm800k_jsonl"
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = "tasksource/PRM800K"

    # Map split names to files
    split_files = {
        "train": ["phase1_train.jsonl", "phase2_train.jsonl"],
        "test": ["phase1_test.jsonl", "phase2_test.jsonl"],
        "all": ["phase1_train.jsonl", "phase2_train.jsonl", "phase1_test.jsonl", "phase2_test.jsonl"]
    }

    if split not in split_files:
        raise ValueError(f"split must be 'train', 'test', or 'all'. Got: {split}")

    files_to_load = split_files[split]

    print(f"📥 Loading PRM800K ({split} split)...")

    # Download and load files
    datasets_dict = {}

    for filename in files_to_load:
        # Determine the split name for this file
        if "train" in filename:
            split_name = "train"
        else:
            split_name = "test"

        if split_name not in datasets_dict:
            datasets_dict[split_name] = []

        print(f"   📄 Downloading {filename}...")
        try:
            filepath = hf_hub_download(
                repo_id=dataset_id,
                filename=filename,
                repo_type="dataset",
                cache_dir=str(cache_dir),
                force_download=False
            )

            print(f"   📖 Loading {filename}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                count = 0
                for line in f:
                    if line.strip():
                        try:
                            example = json.loads(line)
                            datasets_dict[split_name].append(example)
                            count += 1
                            if limit and count >= limit:
                                break
                        except json.JSONDecodeError:
                            pass

                print(f"      ✅ Loaded {count} examples")

        except Exception as e:
            print(f"      ⚠️  Error loading {filename}: {e}")

    # Convert to lists
    result = {}
    for split_name, examples in datasets_dict.items():
        if examples:
            result[split_name] = examples

    print(f"\n✅ Dataset loaded!")

    # Return format
    if split == "all":
        return result
    else:
        return result.get(split)


def extract_step_data(item: Dict) -> List[Dict]:
    """
    Extract step-by-step reasoning data with ratings.
    Output format: problem + partial solution -> next step rating
    """
    examples = []

    problem = item['question']['problem']
    steps = item['label']['steps']
    ground_truth = item['question'].get('ground_truth_answer', '')

    # Build conversation incrementally
    conversation_history = []

    for step_idx, step in enumerate(steps):
        if step['chosen_completion'] is None:
            continue

        chosen_idx = step['chosen_completion']
        chosen_step = step['completions'][chosen_idx]

        # Create the prompt with conversation history
        if len(conversation_history) == 0:
            prompt = f"Problem: {problem}\n\nSolution steps so far:\n"
        else:
            prompt = f"Problem: {problem}\n\nSolution steps so far:\n"
            for i, prev_step in enumerate(conversation_history, 1):
                prompt += f"Step {i}: {prev_step}\n"

        prompt += f"\nStep {len(conversation_history) + 1}: {chosen_step['text']}"

        # Rating: -1 (bad), 0 (neutral), 1 (good)
        rating = chosen_step['rating']
        rating_text = {
            -1: "incorrect or unhelpful",
            0: "neutral or partially correct",
            1: "correct and helpful"
        }.get(rating, "unknown")

        # Create label with rating
        label = f"Rating: {rating} ({rating_text})"
        if chosen_step.get('flagged', False):
            label += " [FLAGGED]"

        examples.append({
            'text': prompt,
            'label': label,
            'rating': rating,
            'step_number': len(conversation_history) + 1,
            'total_steps': len(steps),
            'problem': problem,
            'ground_truth': ground_truth
        })

        # Add chosen step to history
        conversation_history.append(chosen_step['text'])

        # Include negative examples (unchosen completions)
        for comp_idx, completion in enumerate(step['completions']):
            if comp_idx != chosen_idx and completion.get('rating') is not None and completion['rating'] <= 0:
                negative_prompt = prompt.replace(
                    chosen_step['text'],
                    completion['text']
                )

                neg_rating_text = {
                    -1: "incorrect or unhelpful",
                    0: "neutral or partially correct",
                    1: "correct and helpful"
                }.get(completion['rating'], "unknown")

                negative_label = f"Rating: {completion['rating']} ({neg_rating_text})"
                if completion.get('flagged', False):
                    negative_label += " [FLAGGED]"

                examples.append({
                    'text': negative_prompt,
                    'label': negative_label,
                    'rating': completion['rating'],
                    'step_number': len(conversation_history),
                    'total_steps': len(steps),
                    'problem': problem,
                    'ground_truth': ground_truth,
                    'is_negative': True
                })

    return examples


def format_for_training(example: Dict, include_label: bool = True) -> str:
    """Format example for training"""
    if include_label:
        return f"{example['text']}\n\n{example['label']}"
    else:
        return example['text']


def prepare_dataset(data: List[Dict], output_file: str = None,
                   include_negatives: bool = True,
                   max_samples: int = None):
    """
    Prepare the complete dataset for training.
    """
    all_examples = []

    print("Processing PRM800K data...")
    for item in tqdm(data):
        examples = extract_step_data(item)

        if not include_negatives:
            # Filter out negative examples
            examples = [ex for ex in examples if not ex.get('is_negative', False)]

        all_examples.extend(examples)

    print(f"\nTotal examples extracted: {len(all_examples)}")

    # Shuffle
    random.shuffle(all_examples)

    # Limit samples if specified
    if max_samples:
        all_examples = all_examples[:max_samples]
        print(f"Limited to {max_samples} samples")

    # Format for training
    formatted_data = []
    for ex in all_examples:
        formatted_data.append({
            'input_text': format_for_training(ex, include_label=False),
            'output_text': ex['label'],
            'full_text': format_for_training(ex, include_label=True),
            'rating': ex['rating'],
            'step_number': ex['step_number'],
            'total_steps': ex['total_steps']
        })

    # Save if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(formatted_data, f)
        print(f"Saved to {output_file}")

    return formatted_data


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("PRM800K DATA PREPROCESSING (from Verifier.ipynb)")
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
    train_ratings = Counter([ex.get('rating') for ex in train_formatted])
    print(f"\nRating distribution (train):")
    for rating in sorted([r for r in train_ratings.keys() if r is not None]):
        count = train_ratings[rating]
        pct = 100 * count / len(train_formatted)
        print(f"  Rating {rating}: {count} ({pct:.1f}%)")
    
    # Check for None ratings
    none_count = train_ratings.get(None, 0)
    if none_count > 0:
        print(f"  Rating None: {none_count} ({100 * none_count / len(train_formatted):.1f}%)")
    
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