import os
from pathlib import Path

# Define paths
data_dir = Path("data")
train_dir = data_dir / "train"
test_dir = data_dir / "test"

# Emotion classes
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Count samples for each emotion
results = []

print("=" * 80)
print("EMOTION RECOGNITION MODEL - DATASET ANALYSIS")
print("=" * 80)
print()

for emotion in emotions:
    train_path = train_dir / emotion
    test_path = test_dir / emotion
    
    # Count files
    train_count = len(list(train_path.glob("*.jpg"))) if train_path.exists() else 0
    test_count = len(list(test_path.glob("*.jpg"))) if test_path.exists() else 0
    total_count = train_count + test_count
    
    train_pct = f"{(train_count / total_count * 100):.1f}%" if total_count > 0 else "0%"
    test_pct = f"{(test_count / total_count * 100):.1f}%" if total_count > 0 else "0%"
    
    results.append({
        "emotion": emotion.upper(),
        "train": train_count,
        "test": test_count,
        "total": total_count,
        "train_pct": train_pct,
        "test_pct": test_pct
    })

# Print table header
print(f"{'EMOTION':<12} {'TRAINING':<12} {'TESTING':<12} {'TOTAL':<12} {'RATIO':<20}")
print("-" * 80)

# Print each row
for row in results:
    print(f"{row['emotion']:<12} {row['train']:<12} {row['test']:<12} {row['total']:<12} {row['train_pct']} train, {row['test_pct']} test")

print()
print("=" * 80)

# Summary statistics
total_train = sum(r["train"] for r in results)
total_test = sum(r["test"] for r in results)
total_all = total_train + total_test

print(f"\nTOTAL STATISTICS:")
print(f"  Total Training Samples: {total_train:,}")
print(f"  Total Testing Samples:  {total_test:,}")
print(f"  Grand Total:            {total_all:,}")
print(f"  Train/Test Ratio:       {(total_train/total_all*100):.1f}% train, {(total_test/total_all*100):.1f}% test")
print()

# Check class balance
totals = [r["total"] for r in results]
min_samples = min(totals)
max_samples = max(totals)
avg_samples = total_all / len(emotions)

print("CLASS BALANCE ANALYSIS:")
print(f"  Samples per class range: {min_samples:,} - {max_samples:,}")
print(f"  Average per class:       {avg_samples:.0f}")
print(f"  Class imbalance ratio:   {max_samples/min_samples:.2f}x")
print()

# Recommendations
print("RECOMMENDATIONS:")
imbalance = max(totals) - min(totals)
if imbalance > avg_samples * 0.3:
    print("  ⚠ WARNING: Significant class imbalance detected!")
    print("    Consider using class weighting or data augmentation.")
else:
    print("  ✓ Good balance across emotion classes")

print("=" * 80)
