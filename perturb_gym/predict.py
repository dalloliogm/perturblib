"""Copyright (C) 2025  GlaxoSmithKline plc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Module for making predictions with trained perturbation models and saving to AnnData.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import torch

import perturb_lib as plib
from perturb_gym.configs.access import load_training_configs
from perturb_gym.configs.base import DataConfig, EnvironmentConfig, ModelConfig, TrainingConfig


def predict_from_trained_model(
    model_path: str | Path,
    prediction_context: str,
    output_path: str | Path,
    model_config: Optional[ModelConfig] = None,
    batch_size: int = 1000,
) -> None:
    """Make predictions using a trained model and save results to AnnData.

    Args:
        model_path: Path to the trained model file (.pt)
        prediction_context: Context identifier for the prediction dataset
        output_path: Path where to save the predictions as AnnData
        model_config: Model configuration (if None, will try to infer from model)
        batch_size: Batch size for prediction
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    # Load the trained model
    if model_config is None:
        # Try to infer model config from the model path structure
        # This is a simplified approach - in practice you might need more sophisticated config loading
        model_config = ModelConfig(
            model_id="LPM",
            model_args={
                "optimizer_name": "AdamW",
                "learning_rate": 0.002,
                "learning_rate_decay": 0.99,
                "num_layers": 2,
                "hidden_dim": 512,
                "dropout": 0.0,
                "batch_size": 5000,
                "embedding_dim": 32,
            }
        )

    model = plib.load_trained_model(model_path, model_config.model_args)

    # Load prediction data
    plib.logger.info(f"Loading prediction data from context: {prediction_context}")
    pred_adata = plib.load_anndata(prediction_context)

    # Convert to PlibData format for prediction
    pred_data = plib.load_plibdata(prediction_context, plibdata_type=plib.InMemoryPlibData)

    # Make predictions
    plib.logger.info("Making predictions...")
    predictions = model.predict(pred_data, batch_size=batch_size)

    # Convert predictions back to AnnData format
    plib.logger.info("Converting predictions to AnnData format...")

    # Create a copy of the original adata for predictions
    pred_adata_copy = pred_adata.copy()

    # The predictions should be in the same shape as the original data
    # predictions is likely a numpy array or tensor
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    # Replace the X matrix with predictions
    pred_adata_copy.X = predictions

    # Add metadata to indicate these are predictions
    pred_adata_copy.uns["prediction_info"] = {
        "model_path": str(model_path),
        "prediction_context": prediction_context,
        "model_config": model_config.model_args,
    }

    # Save the predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plib.logger.info(f"Saving predictions to {output_path}")
    pred_adata_copy.write_h5ad(output_path)

    plib.logger.info("Prediction complete!")


def predict_from_config_file(
    config_file_id_or_path: str,
    prediction_context: str,
    output_dir: str | Path,
    model_seed: int = 13,
    results_dir: Optional[str | Path] = None,
) -> None:
    """Make predictions using models trained from a config file.

    Args:
        config_file_id_or_path: Config file identifier or path
        prediction_context: Context for prediction data
        output_dir: Directory to save predictions
        model_seed: Which seed/model to use for prediction
        results_dir: Directory where trained models are stored
    """
    output_dir = Path(output_dir)

    if results_dir is None:
        results_dir = plib.get_path_to_cache() / "results"
    else:
        results_dir = Path(results_dir)

    # Load training configs to find the model
    training_configs = load_training_configs(config_file_id_or_path)

    # Find the config with the matching seed
    target_config = None
    for config in training_configs:
        if config.environment_config.seed == model_seed:
            target_config = config
            break

    if target_config is None:
        raise ValueError(f"No training config found with seed {model_seed}")

    # Construct model path
    unique_model_name = (
        f"{target_config.model_config.model_id}_"
        f"{plib.training.hash_training_config_excluding_seed(target_config)}"
    )
    model_dir = results_dir / config_file_id_or_path / unique_model_name / f"seed_{model_seed}"
    model_path = model_dir / "model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    # Create output path
    output_path = output_dir / f"predictions_{config_file_id_or_path}_seed_{model_seed}.h5ad"

    # Make predictions
    predict_from_trained_model(
        model_path=model_path,
        prediction_context=prediction_context,
        output_path=output_path,
        model_config=target_config.model_config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with trained perturbation models")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Config file identifier or path"
    )
    parser.add_argument(
        "--prediction_context",
        type=str,
        required=True,
        help="Context identifier for prediction data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./predictions",
        help="Directory to save predictions"
    )
    parser.add_argument(
        "--model_seed",
        type=int,
        default=13,
        help="Model seed to use for prediction"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory where trained models are stored"
    )

    args = parser.parse_args()

    predict_from_config_file(
        config_file_id_or_path=args.config_file,
        prediction_context=args.prediction_context,
        output_dir=args.output_dir,
        model_seed=args.model_seed,
        results_dir=args.results_dir,
    )