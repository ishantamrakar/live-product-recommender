import datasets
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import json

datasets.logging.set_verbosity_error()

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
N_REVIEWS = 200_000
SHUFFLE_BUFFER = 50_000
SEED = 42


def download_reviews():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_file = DATA_DIR / "reviews_electronics.jsonl"
    if output_file.exists():
        print(f"✓ Reviews already downloaded: {output_file}")
        return

    print(f"Streaming {N_REVIEWS:,} Electronics reviews (shuffled)...\n")

    stream = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_Electronics",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )
    stream = stream.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)

    saved = 0
    with open(output_file, 'w') as f:
        with tqdm(total=N_REVIEWS, unit="reviews") as pbar:
            for review in stream:
                f.write(json.dumps(review) + '\n')
                saved += 1
                pbar.update(1)
                if saved >= N_REVIEWS:
                    break

    print(f"✓ Saved {saved:,} reviews\n")


def download_metadata():
    output_file = DATA_DIR / "meta_electronics.jsonl"

    if output_file.exists():
        print(f"✓ Metadata already downloaded: {output_file}")
        return

    print("Streaming full Electronics metadata (no filter)...")
    stream = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Electronics",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    saved = 0
    with open(output_file, 'w') as f:
        with tqdm(unit="products") as pbar:
            for item in stream:
                out_item = {k: v for k, v in item.items() if k != 'images'}
                f.write(json.dumps(out_item) + '\n')
                saved += 1
                pbar.update(1)

    print(f"✓ Saved {saved:,} products\n")


def peek(file, n=3):
    print("="*80)
    print(f"Sample from {file.name}:")
    print("="*80)
    with open(file) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            record = json.loads(line)
            for k, v in record.items():
                if isinstance(v, str) and len(v) > 120:
                    v = v[:120] + "..."
                if v not in (None, [], {}):
                    print(f"  {k}: {v}")
            print()


if __name__ == "__main__":
    download_reviews()
    download_metadata()

    print("\nPreviewing data:\n")
    peek(DATA_DIR / "reviews_electronics.jsonl")
    peek(DATA_DIR / "meta_electronics.jsonl")
