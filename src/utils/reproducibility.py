"""
Reproducibility utility for setting global random seeds across random, numpy, and PyTorch.
Enforces deterministic behavior in modeling experiments.
"""

import logging
import random
import numpy as np

logger = logging.getLogger(__name__)

def set_seeds(seed: int = 42, deterministic: bool = True) -> None:
    """
    Sets the random seed for python's built-in random, numpy, and torch (if installed)
    to ensure reproducible results.
    
    Args:
        seed (int): The seed number to set.
        deterministic (bool): If True, configures PyTorch to use deterministic algorithms 
                              where possible, at the cost of potential performance degradation.
    """
    logger.info(f"Setting global random seed to {seed} (deterministic={deterministic})")
    
    # 1. Standard Python random
    random.seed(seed)
    
    # 2. NumPy random
    np.random.seed(seed)
    
    # 3. PyTorch (optional import to avoid strict dependency crashes if torch isn't fully set up yet)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # For multi-GPU configurations
            
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Modern PyTorch deterministic flag
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception as ex:
                    logger.debug(f"Could not enable strict deterministic algorithms: {ex}")
        logger.debug("Successfully seeded PyTorch and CUDA configurations.")
    except ImportError:
        logger.debug("PyTorch is not installed or could not be imported. Seeding skipped for torch.")
    except Exception as e:
        logger.warning(f"Unexpected error when configuring PyTorch reproducibility: {e}")
