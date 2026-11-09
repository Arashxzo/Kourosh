"""
Kourosh - Download Real Persian Data
=====================================
Streams Persian Wikipedia from HuggingFace and saves a fixed-size slice
to data/raw/corpus.txt (overwrites the demo corpus).

Usage:
    python -m src.download_data --mb 10
"""
import argparse
import os
import sys

from src.config import RAW_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mb", type=int, default=10, help="Target size in MB")
    parser.add_argument("--dataset", type=str, default="wikimedia/wikipedia",
                        help="HuggingFace dataset")
    parser.add_argument("--config", type=str, default="20231101.fa",
                        help="Dataset config (Persian Wikipedia)")
    args = parser.parse_args()

    target_bytes = args.mb * 1024 * 1024

    # Force HF mirror (works from Iran)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"HF_ENDPOINT = {os.environ['HF_ENDPOINT']}")

    print(f"Streaming {args.dataset}/{args.config} ...")
    print(f"Target: {args.mb} MB → {RAW_DATA_DIR / 'corpus.txt'}\n")

    try:
        from datasets import load_dataset
        ds = load_dataset(
            args.dataset,
            args.config,
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"\n[ERROR] Could not load dataset: {e}")
        print("\nFallback: try a smaller slice or a different mirror.")
        sys.exit(1)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / "corpus.txt"

    written = 0
    n_articles = 0

    with out_path.open("w", encoding="utf-8") as f:
        for article in ds:
            text = article.get("text", "").strip()
            if len(text) < 300:
                continue

            text = " ".join(text.split())
            if len(text) > 15000:
                text = text[:15000]

            f.write(text + "\n\n")
            written += len(text.encode("utf-8"))
            n_articles += 1

            if n_articles % 200 == 0:
                pct = 100 * written / target_bytes
                print(f"  {n_articles:5d} articles | {written/1e6:.2f} MB | {pct:.0f}%")

            if written >= target_bytes:
                break

    size_mb = out_path.stat().st_size / 1e6
    print(f"\n✅ Done. {n_articles:,} articles, {size_mb:.2f} MB → {out_path}")
    print("Next: python -m src.train")


if __name__ == "__main__":
    main()