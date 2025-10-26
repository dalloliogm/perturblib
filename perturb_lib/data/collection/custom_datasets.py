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

Loading and processing of custom perturbation datasets.
"""

import numpy as np
import scanpy as sc
from pathlib import Path

from perturb_lib._utils import download_file
from perturb_lib.data import ControlSymbol
from perturb_lib.data.access import create_and_register_context
from perturb_lib.environment import get_path_to_cache, logger


def _get_arc_challenge_train_dataset():
    """Load ARC challenge training dataset."""
    path = Path("../state/data/competition_support_set/competition_train.h5")
    if not path.exists():
        raise FileNotFoundError(f"Custom dataset not found at {path}")

    logger.info(f"Loading custom dataset from {path}")
    adata = sc.read_h5ad(path)

    # Standardize the format if needed
    if "gene" in adata.obs.columns:
        adata.obs = adata.obs.rename(columns={"gene": "perturbation_target"})
    if "gene_name" in adata.var.columns:
        adata.var = adata.var.rename(columns={"gene_name": "readout_target"})

    # Ensure required columns exist
    if "perturbation_target" not in adata.obs.columns:
        raise ValueError("Dataset must have 'perturbation_target' column in obs")
    if "readout_target" not in adata.var.columns:
        raise ValueError("Dataset must have 'readout_target' column in var")

    # Add required metadata
    adata.obs["perturbation_type"] = "CRISPRi"
    adata.obs = adata.obs[["perturbation_target", "perturbation_type"]]
    adata.var["readout_type"] = "Transcriptome"
    adata.var = adata.var[["readout_target", "readout_type"]]

    # Type conversion
    adata.obs.index = adata.obs.index.astype(str)
    adata.var.index = adata.var.index.astype(str)

    # Handle infinite values
    infinite_indices = np.where(~np.isfinite(adata.X))
    if len(infinite_indices[0]) > 0:
        logger.debug(f"Zero-imputation of {len(infinite_indices[0])} values..")
        adata.X[infinite_indices] = 0.0

    return adata.X, adata.obs, adata.var


def _get_arc_challenge_val_dataset():
    """Load ARC challenge validation dataset."""
    path = Path("../state/data/competition_support_set/competition_val_template.h5")
    if not path.exists():
        raise FileNotFoundError(f"Custom dataset not found at {path}")

    logger.info(f"Loading custom dataset from {path}")
    adata = sc.read_h5ad(path)

    # Standardize the format if needed
    if "gene" in adata.obs.columns:
        adata.obs = adata.obs.rename(columns={"gene": "perturbation_target"})
    if "gene_name" in adata.var.columns:
        adata.var = adata.var.rename(columns={"gene_name": "readout_target"})

    # Ensure required columns exist
    if "perturbation_target" not in adata.obs.columns:
        raise ValueError("Dataset must have 'perturbation_target' column in obs")
    if "readout_target" not in adata.var.columns:
        raise ValueError("Dataset must have 'readout_target' column in var")

    # Add required metadata
    adata.obs["perturbation_type"] = "CRISPRi"
    adata.obs = adata.obs[["perturbation_target", "perturbation_type"]]
    adata.var["readout_type"] = "Transcriptome"
    adata.var = adata.var[["readout_target", "readout_type"]]

    # Type conversion
    adata.obs.index = adata.obs.index.astype(str)
    adata.var.index = adata.var.index.astype(str)

    # Handle infinite values
    infinite_indices = np.where(~np.isfinite(adata.X))
    if len(infinite_indices[0]) > 0:
        logger.debug(f"Zero-imputation of {len(infinite_indices[0])} values..")
        adata.X[infinite_indices] = 0.0

    return adata.X, adata.obs, adata.var


# Register the custom contexts
create_and_register_context(
    model_system="HumanCellLine",
    model_system_id=None,
    technology_info="CRISPRi",
    data_source_info="ARCChallenge",
    batch_info="Train",
    full_context_description="Custom ARC challenge training dataset for perturbation modeling.",
    anndata_fn=_get_arc_challenge_train_dataset,
)

create_and_register_context(
    model_system="HumanCellLine",
    model_system_id=None,
    technology_info="CRISPRi",
    data_source_info="ARCChallenge",
    batch_info="Val",
    full_context_description="Custom ARC challenge validation dataset for perturbation modeling.",
    anndata_fn=_get_arc_challenge_val_dataset,
)