import os
import csv
import torch
import pandas as pd
from typing import Optional, Union
import logging
import datetime


logger = logging.getLogger(__name__)


def set_hardware_acceleration(default: Optional[str] = None) -> torch.device:
    if default is not None:
        device = torch.device(default)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(
                f"There are {torch.cuda.device_count()} GPUs available. Using the {torch.cuda.get_device_name()} GPU."
            )
        else:
            device = torch.device("cpu")
            logger.info("No GPUs available, using CPU instead.")
    return device


def format_time(seconds: Union[int, float]) -> str:
    formatted_time = str(datetime.timedelta(seconds=round(seconds)))
    return formatted_time


def gpu_memory_usage() -> Optional[pd.DataFrame]:
    if torch.cuda.is_available():
        results = os.popen('nvidia-smi --query-gpu=memory.total,memory.used --format=csv')
        reader = csv.reader(results, delimiter=",")
        df = pd.DataFrame(reader)
        df.columns = df.iloc[0]
        df = df[1:]
        return df
    else:
        logger.warning("You called gpu_memory_usage but no GPU is available.")
