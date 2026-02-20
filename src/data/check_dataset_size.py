import datasets
from datasets import load_dataset, load_dataset_builder

datasets.logging.set_verbosity_error()

print("Checking Amazon Reviews 2023 - Electronics dataset size...\n")

for config, label in [("raw_review_Electronics", "Reviews"), ("raw_meta_Electronics", "Metadata")]:
    try:
        builder = load_dataset_builder(
            "McAuley-Lab/Amazon-Reviews-2023",
            config
        )
        info = builder.info
        size_gb = (info.dataset_size or 0) / (1024 ** 3)
        splits = {k: v.num_examples for k, v in info.splits.items()} if info.splits else "unknown"
        print(f"{label}:")
        print(f"  Size:   {size_gb:.2f} GB")
        print(f"  Splits: {splits}\n")
    except Exception as e:
        print(f"{label}: could not fetch builder info — {e}\n")
