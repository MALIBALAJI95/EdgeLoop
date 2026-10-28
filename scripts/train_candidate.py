#!/usr/bin/env python3
"""EdgeLoop Candidate Training & Promotion Evaluation CLI.

Trains a new candidate MobileNetV2 model on verified ground truth data,
registers it in the model registry, and evaluates it against the active
champion using strict statistical governance gates.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path
import pandas as pd

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgeloop.common.database import get_db
from edgeloop.registry.manager import ModelRegistry
from edgeloop.training.trainer import CandidateTrainer


def bootstrap_edge_samples(db, count: int = 8):
    """Seed verified samples from edge research corrupted sets if database has few samples."""
    print(f"📦 Bootstrapping {count} verified edge samples for candidate fine-tuning...")
    seeds = [
        ("research/edgeloop/corrupted_edge/low_light/plastic_plastic418.jpg", "plastic", "seed_edge_plastic_1"),
        ("research/edgeloop/corrupted_edge/low_light/plastic_plastic168.jpg", "plastic", "seed_edge_plastic_2"),
        ("research/edgeloop/corrupted_edge/low_light/paper_paper172.jpg", "paper", "seed_edge_paper_1"),
        ("research/edgeloop/corrupted_edge/low_light/cardboard_cardboard242.jpg", "cardboard", "seed_edge_cardboard_1"),
        ("research/edgeloop/corrupted_edge/blur/metal_metal29.jpg", "metal", "seed_edge_metal_1"),
        ("research/edgeloop/corrupted_edge/blur/glass_glass1.jpg", "glass", "seed_edge_glass_1"),
        ("research/edgeloop/corrupted_edge/noise/glass_glass1.jpg", "glass", "seed_edge_glass_2"),
        ("research/edgeloop/corrupted_edge/low_light/paper_paper71.jpg", "trash", "seed_edge_trash_1"),
    ]

    save_dir = Path("data/reviewed")
    save_dir.mkdir(parents=True, exist_ok=True)

    for p_str, true_lbl, sid in seeds[:count]:
        src_p = Path(p_str)
        if src_p.exists():
            dst_p = save_dir / f"{sid}.jpg"
            shutil.copy2(src_p, dst_p)
            inf_id = db.log_inference(sid, str(dst_p.resolve()), true_lbl, 0.90, 0.20, 35.0, "champion", "bootstrap_seed", uploaded=1)
            q_id = db.add_to_selection_queue(inf_id, sid, str(dst_p.resolve()), "seed", "bootstrap", 0.20)
            db.submit_review(q_id, inf_id, sid, str(dst_p.resolve()), true_lbl, true_lbl, "system_seed", "Bootstrapped verified sample")

    print(f"✅ Bootstrapped verified samples committed to ground truth registry.")


def main():
    parser = argparse.ArgumentParser(description="EdgeLoop Candidate Trainer")
    parser.add_argument("--epochs", type=int, default=2, help="Fine-tuning epochs (default: 2)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--version", type=str, default="", help="Candidate version name")
    parser.add_argument("--eval-samples", type=int, default=100, help="Holdout evaluation sample count (default: 100)")
    args = parser.parse_args()

    cand_ver = args.version or f"candidate_edge_{int(time.time())}"
    print("=================================================================")
    print("🚀 EdgeLoop Continuous Learning & Candidate Retraining")
    print(f"Candidate Version: {cand_ver}")
    print(f"Epochs:            {args.epochs} | Learning Rate: {args.lr}")
    print("=================================================================\n")

    db = get_db()
    reg = ModelRegistry()

    # Load verified samples
    with db.get_connection() as conn:
        df_verified = pd.read_sql("SELECT sample_id, image_path, true_label as label FROM review_records", conn)

    if len(df_verified) < 4:
        print(f"⚠️ Only {len(df_verified)} verified samples in registry (minimum required: 4).")
        bootstrap_edge_samples(db)
        with db.get_connection() as conn:
            df_verified = pd.read_sql("SELECT sample_id, image_path, true_label as label FROM review_records", conn)

    print(f"📚 Training Dataset: {len(df_verified)} verified ground truth samples")
    print("   Class Breakdown:")
    for lbl, count in df_verified["label"].value_counts().items():
        print(f"   • {lbl.upper():10s}: {count}")

    # Train Candidate
    print("\n⏳ Initializing MobileNetV2 fine-tuning with leak-free group split...")
    trainer = CandidateTrainer()
    ckpt_path, history = trainer.train_candidate(
        dataset_df=df_verified,
        candidate_name=cand_ver,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    print(f"✅ Candidate weights saved to: {ckpt_path}")

    # Register candidate
    print(f"📝 Registering candidate '{cand_ver}' in model registry...")
    cand_record = reg.register_candidate(version=cand_ver, checkpoint_path=ckpt_path)

    # Evaluate Candidate vs Champion
    print("\n⚖️ Running Governance Holdout Gate Evaluation (Candidate vs Active Champion)...")
    comp = reg.evaluate_candidate(
        candidate_version=cand_ver,
        eval_manifest_path="research/edgeloop/dataset_manifest.csv",
        max_samples=args.eval_samples
    )

    print("\n=================================================================")
    print("📋 GOVERNANCE SCORECARD:")
    print("=================================================================")
    print(f"Champion Version:       {comp.champion_version}")
    print(f"Candidate Version:      {comp.candidate_version}")
    print(f"Champion Macro-F1:      {comp.champion_metrics.macro_f1:.4f} | Accuracy: {comp.champion_metrics.accuracy:.4f}")
    print(f"Candidate Macro-F1:     {comp.candidate_metrics.macro_f1:.4f} | Accuracy: {comp.candidate_metrics.accuracy:.4f}")
    print(f"Macro-F1 Delta:         {comp.macro_f1_delta:+.4f}")
    print(f"Accuracy Delta:         {comp.accuracy_delta:+.4f}")
    print(f"Sensitive Trash Delta:  {comp.trash_f1_delta:+.4f}")

    print("\nPer-Class F1 Breakdown:")
    classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    for c in classes:
        cand_f1 = comp.candidate_metrics.per_class.get(c).f1_score if c in comp.candidate_metrics.per_class else 0.0
        champ_f1 = comp.champion_metrics.per_class.get(c).f1_score if c in comp.champion_metrics.per_class else 0.0
        print(f"   • {c.upper():10s} -> Champion: {champ_f1:.4f} | Candidate: {cand_f1:.4f} | Delta: {cand_f1 - champ_f1:+.4f}")

    print("\nGate Verification:")
    for gate_name, passed in comp.gate_checks.items():
        symbol = "✅ PASS" if passed else "❌ FAIL"
        print(f"   [{symbol}] {gate_name}")

    if comp.passes_all_gates:
        print("\n🏆 PROMOTION CRITERIA MET! Candidate has passed all governance gates.")
        print(f"To promote candidate to active champion, run:")
        print(f"  .venv/bin/python -c \"from edgeloop.registry.manager import ModelRegistry; ModelRegistry().promote_candidate('{cand_ver}', approver='MLOps Engineer', promotion_reason='Passed holdout gate')\"")
    else:
        print("\n🛡️ GOVERNANCE ENFORCEMENT: Candidate did not satisfy promotion criteria.")
        print("Reasons:")
        for r in comp.reasons:
            print(f"   - {r}")
        print("Active Champion remains retained in production.")
    print("=================================================================")


if __name__ == "__main__":
    main()
