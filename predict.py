import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from model import predict_flower, CATEGORIES


def main():
    parser = argparse.ArgumentParser(
        description="ScratchNet Flower Classifier — CLI Image Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py predict.py test_images/download.jpg
  py predict.py path/to/rose.png
        """,
    )
    parser.add_argument("image_path", type=str, help="Path to the flower image file.")
    args = parser.parse_args()

    img_path = Path(args.image_path)
    if not img_path.is_file():
        print(f"Error: File not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nAnalyzing: {img_path.name} ...")
    try:
        result = predict_flower(img_path)
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        sys.exit(1)

    winner = result["top_flower"]
    conf = result["confidence_pct"]
    device = result["device"]
    info = result["flower_info"]

    print("=" * 50)
    print(f"  {info.get('icon', '🌸')} Top Prediction: {winner.upper()} ({conf})")
    print(f"  Botanical Name: {info.get('scientific_name', 'N/A')}")
    print(f"  Hardware:       {device.upper()}")
    print("-" * 50)
    print("  Full Probability Spectrum:")
    for item in result["ranked_list"]:
        bar = "█" * int(item["percent_value"] / 5)
        print(f"   {item['icon']} {item['display_name']:12s} {item['percentage']:>6s} | {bar}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
