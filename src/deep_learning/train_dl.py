"""
Deep learning training orchestration for longitudinal voice biomarkers.
Constructs temporal sequence datasets from patient histories and trains
a PyTorch LSTM progression regression model.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from src.utils.config_loader import resolve_path
from src.modeling.train import group_train_test_split, compute_metrics

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from src.deep_learning.models import BiomarkerLSTM
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class DeepLearningError(Exception):
    """Custom exception raised for deep learning pipeline errors."""
    pass

if TORCH_AVAILABLE:
    class LongitudinalVoiceDataset(Dataset):
        """
        PyTorch Dataset for structuring longitudinal voice recordings into sequence windows.
        Pipes sequential patient entries into a recurrent sequence model.
        """
        def __init__(
            self, 
            df: pd.DataFrame, 
            feature_cols: List[str], 
            target_col: str, 
            subject_col: str, 
            seq_len: int = 5
        ):
            """
            Args:
                df (pd.DataFrame): Dataframe containing longitudinal patient records.
                feature_cols (List[str]): Input biomarker column names.
                target_col (str): Prediction target column.
                subject_col (str): Column mapping individual subjects.
                seq_len (int): Window sequence size for recurrent learning.
            """
            self.seq_len = seq_len
            self.features = []
            self.targets = []
            
            # Group by subject and extract sequential windows
            for _, group in df.groupby(subject_col):
                # Sort chronological
                group_sorted = group.sort_values(by="test_time")
                
                # Check if group has enough visits
                if len(group_sorted) < seq_len:
                    # Pad sequence with first visit duplicated if too short
                    n_needed = seq_len - len(group_sorted)
                    pad_rows = pd.concat([group_sorted.iloc[[0]]] * n_needed, ignore_index=True)
                    padded_group = pd.concat([pad_rows, group_sorted], ignore_index=True)
                    
                    X = padded_group[feature_cols].values
                    y = padded_group[target_col].values[-1] # Target is the last value
                    self.features.append(X)
                    self.targets.append(y)
                else:
                    # Construct rolling window slices
                    for i in range(len(group_sorted) - seq_len + 1):
                        window = group_sorted.iloc[i : i + seq_len]
                        X = window[feature_cols].values
                        y = window[target_col].values[-1] # Predict target at the end of window
                        self.features.append(X)
                        self.targets.append(y)
                        
            self.features = np.array(self.features, dtype=np.float32)
            self.targets = np.array(self.targets, dtype=np.float32)
            
        def __len__(self) -> int:
            return len(self.features)
            
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.tensor(self.features[idx]), 
                torch.tensor(self.targets[idx]).unsqueeze(-1)
            )
else:
    # Fallback dummy class
    class LongitudinalVoiceDataset:
        """Placeholder class for PyTorch Dataset."""
        def __init__(self, *args, **kwargs):
            pass

def run_deep_learning_training(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Orchestrates sequential data construction and PyTorch model training.
    
    Args:
        df (pd.DataFrame): Feature-engineered DataFrame.
        config (Dict[str, Any]): Loaded project configurations.
        
    Returns:
        Dict[str, float]: Metrics dictionary calculated on testing split.
        
    Raises:
        DeepLearningError: If PyTorch runs fail.
    """
    logger.info("Initializing deep learning pipeline...")
    
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch is not available in the current Python environment. Skipping DL training stage.")
        return {"MSE": -1.0, "RMSE": -1.0, "MAE": -1.0, "R2": -1.0}
        
    # Extract configs
    schema_cfg = config["data_validation"]["schema"]
    dl_cfg = config["deep_learning"]
    paths_cfg = config["paths"]
    
    subject_col = schema_cfg["subject_id_col"]
    target_col = schema_cfg["total_updrs_target"]
    
    # Feature columns
    exclude_cols = [
        subject_col, 
        schema_cfg.get("age_col", "age"), 
        schema_cfg.get("sex_col", "sex"), 
        schema_cfg["test_time_col"], 
        schema_cfg["motor_updrs_target"], 
        schema_cfg["total_updrs_target"]
    ]
    features = [col for col in df.columns if col not in exclude_cols]
    
    # 1. Split data (grouped by subject)
    train_df, test_df = group_train_test_split(
        df=df,
        subject_col=subject_col,
        test_size=0.2, # Allocate 20% of subjects to test
        seed=config.get("reproducibility", {}).get("seed", 42)
    )
    
    # 2. Datasets & Dataloaders
    seq_len = dl_cfg.get("sequence_length", 5)
    train_dataset = LongitudinalVoiceDataset(train_df, features, target_col, subject_col, seq_len)
    test_dataset = LongitudinalVoiceDataset(test_df, features, target_col, subject_col, seq_len)
    
    batch_size = dl_cfg.get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Model setup
    device_name = dl_cfg.get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    logger.info(f"Using device for deep learning: {device}")
    
    model = BiomarkerLSTM(
        input_dim=len(features),
        hidden_dim=dl_cfg.get("hidden_dim", 64),
        num_layers=dl_cfg.get("num_layers", 2),
        output_dim=1,
        dropout=dl_cfg.get("dropout", 0.2)
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=dl_cfg.get("learning_rate", 0.001), 
        weight_decay=dl_cfg.get("weight_decay", 1e-4)
    )
    
    # 4. Training loop (we keep epochs low during test/skeleton phase for speed)
    epochs = dl_cfg.get("epochs", 5) # Default config uses 50, but we can override or let it run
    # For initial run verification, we can train for 2 epochs to prove infrastructure works
    logger.info(f"Starting LSTM model training for {epochs} epochs...")
    
    try:
        model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * X_batch.size(0)
                
            avg_loss = epoch_loss / len(train_dataset)
            if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
                logger.info(f"Epoch {epoch}/{epochs} - Train MSE Loss: {avg_loss:.4f}")
                
        # 5. Evaluation
        model.eval()
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch)
                all_preds.append(preds.cpu().numpy())
                all_trues.append(y_batch.numpy())
                
        y_pred = np.vstack(all_preds).squeeze()
        y_true = np.vstack(all_trues).squeeze()
        
        # Calculate metrics
        metrics = compute_metrics(y_true, y_pred)
        
        for name, val in metrics.items():
            logger.info(f"DL Test Metric - {name}: {val:.4f}")
            
        # 6. Save model weights
        models_dir = resolve_path(paths_cfg["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = models_dir / "lstm_progression_weights.pt"
        
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Deep learning weights serialized to: {model_save_path.as_posix()}")
        
        # Save evaluation metrics
        eval_dir = resolve_path(paths_cfg["evaluation_dir"])
        eval_dir.mkdir(parents=True, exist_ok=True)
        metrics_save_path = eval_dir / "dl_metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_save_path, index=False)
        
        return metrics
    except Exception as e:
        raise DeepLearningError(f"Error during deep learning training: {e}")
