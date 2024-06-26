from .utils import set_hardware_acceleration, format_time, gpu_memory_usage
from typing import Optional, Tuple
from tqdm import tqdm
from time import time
import torch
import logging

logger = logging.getLogger(__name__)

def predict(
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        model: torch.nn.Module,
        batch_size: int,
        device_: Optional[str] = None,  # if None, it automatically detects if a GPU is available, if not uses a CPU
        disable_progress_bar: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert input_ids.shape == token_type_ids.shape == attention_masks.shape, "Some input shapes are wrong"

    device = set_hardware_acceleration(default=device_)
    model = model.to(device)
    model.eval()

    pred_start = torch.tensor([], dtype=torch.long, device=device)  # initialising tensors for storing results
    pred_end = torch.tensor([], dtype=torch.long, device=device)

    t_i = time()
    # batch the samples to speed up processing. We do batching manually here to avoid using DataLoader
    for batch_i in tqdm(range(0, len(input_ids), batch_size), disable=disable_progress_bar):
        batch_input_ids = input_ids[batch_i:batch_i + batch_size, :].to(device)
        batch_token_type_ids = token_type_ids[batch_i:batch_i + batch_size, :].to(device)
        batch_attention_masks = attention_masks[batch_i:batch_i + batch_size, :].to(device)
        with torch.no_grad():
            start_logits, end_logits = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
                token_type_ids=batch_token_type_ids,
            )  # if we don't pass it start_positions and end_positions it won't return the loss, unlike during training

            pred_start_positions = torch.argmax(start_logits, dim=1)
            pred_end_positions = torch.argmax(end_logits, dim=1)

            pred_start = torch.cat((pred_start, pred_start_positions))
            pred_end = torch.cat((pred_end, pred_end_positions))
        if torch.cuda.is_available():
            logger.debug(f"GPU memory usage: \n{gpu_memory_usage()}")

    logger.info(f"All predictions calculated in {format_time(time() - t_i)}.")
    if torch.cuda.is_available():
        logger.info(f"GPU memory usage: \n{gpu_memory_usage()}")

    return pred_start, pred_end

