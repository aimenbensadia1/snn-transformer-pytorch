"""
Data loading and generation utilities.

Provides functions for:
- Loading training data from files
- Generating synthetic datasets for experiments
- Creating random training batches
"""

import random
import torch
from typing import Optional

from snn_transformer.config import CONTEXT_SIZE


def load_data(filename: str) -> bytes:
    """
    Load training data from a file as raw bytes.

    Args:
        filename: Path to data file

    Returns:
        Raw bytes content of file
    """
    with open(filename, 'rb') as f:
        return f.read()


def get_random_batch(data: bytes, device: torch.device) -> torch.Tensor:
    """
    Get a random training batch from data.

    Args:
        data: Raw bytes training data
        device: PyTorch device

    Returns:
        Tensor of shape [CONTEXT_SIZE + 1] with input and target tokens
    """
    idx = random.randint(0, len(data) - CONTEXT_SIZE - 2)
    tokens = torch.tensor(
        [data[idx + i] for i in range(CONTEXT_SIZE + 1)],
        dtype=torch.long,
        device=device
    )
    return tokens


def create_toy_dataset(filename: str, size: int = 100000) -> str:
    """
    Create a simple toy dataset with repetitive patterns.

    Good for testing that the model can learn basic patterns.

    Args:
        filename: Output file path
        size: Approximate size in bytes

    Returns:
        The generated text
    """
    patterns = [
        "ABCD" * 25,
        "the quick brown fox jumps over the lazy dog. ",
        "0123456789" * 10,
        "hello world! " * 8,
        "neural network " * 7,
    ]

    data = []
    while len("".join(data)) < size:
        data.append(random.choice(patterns))

    text = "".join(data)[:size]

    with open(filename, 'w') as f:
        f.write(text)

    print(f"Created {filename}: {len(text):,} bytes")
    return text


def create_english_like_data(filename: str, size: int = 500000) -> str:
    """
    Create English-like text data for training.

    Generates synthetic sentences using common English words and
    sentence templates. More realistic than simple patterns but
    still tractable for small models.

    Args:
        filename: Output file path
        size: Approximate size in bytes

    Returns:
        The generated text
    """
    words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
        "is", "are", "was", "were", "been", "being", "has", "had", "does", "did",
        "neural", "network", "learning", "model", "data", "train", "test", "input", "output",
        "layer", "weight", "function", "compute", "memory", "fast", "slow", "large", "small",
    ]

    templates = [
        "{} {} {} {}.",
        "The {} {} {} {} {}.",
        "{} {} {} {} {} {}.",
        "I {} {} {} {}.",
        "We {} {} {} {} {}.",
        "This {} {} {}.",
        "{} {} {} {} and {} {}.",
        "The {} {} {} {} {} {}.",
    ]

    data = []

    while len("".join(data)) < size:
        template = random.choice(templates)
        n_words = template.count("{}")
        sentence_words = [random.choice(words) for _ in range(n_words)]
        sentence = template.format(*sentence_words)

        # Capitalize first letter
        sentence = sentence[0].upper() + sentence[1:]

        data.append(sentence)

        # Occasionally add newline
        if random.random() < 0.3:
            data.append("\n")
        else:
            data.append(" ")

    text = "".join(data)[:size]

    with open(filename, 'w') as f:
        f.write(text)

    print(f"Created {filename}: {len(text):,} bytes")
    return text


def create_mixed_data(filename: str, size: int = 500000) -> str:
    """
    Create mixed difficulty data.

    Combines easy patterns (30%), medium word sequences (40%),
    and harder natural text (30%).

    Args:
        filename: Output file path
        size: Approximate size in bytes

    Returns:
        The generated text
    """
    data = []

    # Easy patterns (30%)
    easy_size = int(size * 0.3)
    patterns = ["ABCD", "0123", "the ", "and "]
    while len("".join(data)) < easy_size:
        data.append(random.choice(patterns) * random.randint(5, 20))
        data.append(" ")

    # Medium patterns (40%)
    medium_words = ["hello", "world", "neural", "network", "machine", "learning"]
    while len("".join(data)) < size * 0.7:
        data.append(" ".join(random.choices(medium_words, k=random.randint(3, 8))))
        data.append(". ")

    # Hard patterns (30%)
    hard_text = """
    The quick brown fox jumps over the lazy dog.
    Machine learning models process data through layers.
    Neural networks learn patterns from examples.
    Transformers use attention mechanisms for sequences.
    Spiking networks use discrete events for computation.
    """
    while len("".join(data)) < size:
        data.append(hard_text)

    text = "".join(data)[:size]

    with open(filename, 'w') as f:
        f.write(text)

    print(f"Created {filename}: {len(text):,} bytes")
    return text


if __name__ == "__main__":
    print("Creating sample datasets...")
    create_toy_dataset("data/toy.txt", 100000)
    create_english_like_data("data/english.txt", 500000)
    create_mixed_data("data/mixed.txt", 500000)
    print("\nDatasets created!")
