#!/usr/bin/env python3
"""CLI utility for single and batch inference with EdgeLoop MobileNetV2."""

import argparse
import json
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgeloop.inference.engine import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description="EdgeLoop Inference CLI")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--dir", type=str, help="Directory of images for batch inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (cpu, mps, cuda, auto)")
    parser.add_argument("--output", type=str, default=None, help="Save predictions to JSON file")
    parser.add_argument("--no-log", action="store_true", help="Do not log predictions to database")

    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("Must provide either --image or --dir")

    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        device=args.device,
        log_predictions=not args.no_log
    )

    results = []

    if args.image:
        print(f"Running inference on: {args.image}")
        res = engine.predict_image(args.image)
        results.append(res.model_dump())
        print(f"Result:")
        print(f"  Predicted Class: {res.predicted_class}")
        print(f"  Confidence:      {res.confidence:.4f}")
        print(f"  Entropy:         {res.entropy:.4f}")
        print(f"  Latency:         {res.inference_time_ms:.2f} ms")
        print(f"  Model Version:   {res.model_version}")

    elif args.dir:
        p = Path(args.dir)
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = [f for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in valid_exts]
        print(f"Running batch inference on {len(images)} images in {args.dir}...")

        batch_res = engine.predict_batch(images)
        for res in batch_res:
            results.append(res.model_dump())

        print(f"Completed batch of {len(batch_res)} images.")
        # Summary class counts
        counts = {}
        for r in batch_res:
            counts[r.predicted_class] = counts.get(r.predicted_class, 0) + 1
        print("Class distribution:", counts)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
