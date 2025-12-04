#!/usr/bin/env bash
set -e

python -m src.data.generate_synthetic_data
python -m src.retrieval.e5_product_encoder
python -m src.features.build_features
python -m src.models.train_two_tower_multimodal
python -m src.models.train_ranker

echo "Pipeline completed successfully."
