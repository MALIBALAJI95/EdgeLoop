# EdgeLoop Data Directory

This directory stores real-time and edge-stream operational data for the EdgeLoop system.

## Directory Structure

- `incoming/`: Stage where edge devices or simulation scripts drop raw images and metadata for inference, feature extraction, drift monitoring, and selective upload.
- `reviewed/`: Verified ground-truth dataset created through the human-in-the-loop review workflow. Only approved records are written here with complete provenance metadata (reviewer, timestamp, original prediction, ground truth, selection reason).
- `reference/`: Calibrated reference distributions and feature baselines for Kolmogorov-Smirnov drift testing.

## Model Governance and Provenance Rules

1. **No Silent Overwrites**: Raw predictions must never overwrite ground-truth data.
2. **Review Barrier**: Unreviewed edge predictions cannot enter training or candidate-evaluation datasets.
3. **Traceability**: Every record stored in `reviewed/` retains its source image identity, edge session ID, inference confidence, uncertainty entropy, drift alert flags, and reviewer attribution.
