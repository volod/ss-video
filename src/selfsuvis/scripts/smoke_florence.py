"""One-frame smoke test for local Florence captioning."""

import argparse
from pathlib import Path

from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a one-frame local Florence smoke test.")
    parser.add_argument("image", help="Path to an image file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from selfsuvis.pipeline.vision.florence import FlorenceModel

    image_path = Path(args.image)
    img = Image.open(image_path).convert("RGB")
    model = FlorenceModel()
    caption, confidence = model.caption_batch([img])[0]
    print(f"image={image_path}")
    print(f"runtime_mode={model.runtime_mode}")
    print(f"caption={caption}")
    print(f"confidence={confidence:.3f}")


if __name__ == "__main__":
    main()
