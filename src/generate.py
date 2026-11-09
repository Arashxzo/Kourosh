"""
Kourosh - Text Generation
==========================
Loads the trained checkpoint and generates text from prompts.
Run after src.train finishes.
"""
import torch

from src.config import DEVICE, CHECKPOINTS_DIR
from src.tokenizer import CharTokenizer
from src.transformer import KouroshLM


def main() -> None:
    ckpt_path = CHECKPOINTS_DIR / "kourosh.pt"
    tok_path = CHECKPOINTS_DIR / "kourosh_tokenizer.json"

    if not ckpt_path.exists():
        raise SystemExit("No checkpoint. Train first: python -m src.train")

    # ---------- Load ----------
    # weights_only=False is required for PyTorch 2.6+ when loading dicts
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    tokenizer = CharTokenizer.load(tok_path)
    model = KouroshLM(**ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("=" * 60)
    print("Kourosh - Text Generation")
    print("=" * 60)
    print(f"Model loaded. Vocab: {tokenizer.vocab_size}\n")

    # ---------- Generate ----------
    # پرامپت‌هایی که در ویکی‌پدیا زیاد یافت می‌شوند
    prompts = [
        "ایران",
        "تهران",
        "کوروش",
        "دانشگاه",
        "کتاب",
    ]

    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([ids], device=DEVICE)
        
        # تولید متن با دمای 0.8 (کمی خلاقانه)
        out = model.generate(
            input_ids,
            max_new_tokens=150,
            temperature=0.8,
            top_k=40,
            eos_id=tokenizer.eos_id,
        )
        text = tokenizer.decode(out[0].tolist())
        
        print(f"--- Prompt: «{prompt}» ---")
        print(text)
        print()


if __name__ == "__main__":
    main()