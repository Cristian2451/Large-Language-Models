from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from .utils import set_hardware_acceleration, format_time, gpu_memory_usage
from typing import Optional, Union, Tuple, Dict
import json
from tqdm import tqdm
from time import time
import torch
import logging


logger = logging.getLogger(__name__)


def _build_dataloaders(
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
        batch_size: Tuple[int, int],
        train_ratio: float = 0.9,
) -> Tuple[DataLoader, DataLoader]:

    dataset = TensorDataset(
        input_ids, token_type_ids, attention_masks, start_positions, end_positions
    )
    train_size = int(train_ratio * len(dataset))
    valid_size = len(dataset) - train_size
    logger.info(
        f"The input dataset has {len(dataset)} input samples, which have been split into {train_size} training "
        f"samples and {valid_size} validation samples."
    )
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size[0], sampler=RandomSampler(train_dataset))  # could do with shuffle=True instead?
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size[1], sampler=SequentialSampler(valid_dataset))
    logger.info(f"There are {len(train_dataloader)} training batches and {len(valid_dataloader)} validation batches.")
    return train_dataloader, valid_dataloader


def fine_tune_train_and_eval(
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
        batch_size: Tuple[int, int],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_ratio: float = 0.9,
        training_epochs: int = 3,
        lr_scheduler_warmup_steps: int = 0,
        save_model_path: Optional[str] = None,
        save_stats_dict_path: Optional[str] = None,
        device_: Optional[str] = None  # if None, it automatically detects if a GPU is available, if not uses a CPU
) -> Tuple[torch.nn.Module, Dict[str, Dict[str, Union[float, str]]]]:

    assert all([isinstance(i, torch.Tensor) for i in [
        input_ids, token_type_ids, attention_masks, start_positions, end_positions
    ]]), "Some inputs are not tensors. When training, start_positions and end_positions must be tensors, not lists."
    assert input_ids.shape == token_type_ids.shape == attention_masks.shape, "Some input shapes are incompatible."
    assert input_ids.shape[0] == len(start_positions) == len(end_positions), "Some input shapes are incompatible"

    train_dataloader, valid_dataloader = _build_dataloaders(
        input_ids, token_type_ids, attention_masks, start_positions, end_positions, batch_size, train_ratio
    )
    training_steps = training_epochs * len(train_dataloader)  # epochs * number of batches
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=lr_scheduler_warmup_steps, num_training_steps=training_steps
    )
    device = set_hardware_acceleration(default=device_)
    model = model.to(device)
    training_stats = {}
    for epoch in (range(training_epochs)):
        logger.info(f"Training epoch {epoch + 1} of {training_epochs}. Running training.")
        t_i = time()
        model.train()
        cumulative_train_loss_per_epoch = 0.
        for batch_num, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            logger.debug(f"Running training batch {batch_num + 1} of {len(train_dataloader)}.")
            batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_start_positions, batch_end_positions = \
                batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
            model.zero_grad()
            #  model.zero_grad() and optimizer.zero_grad() are the same IF all model parameters are in that optimizer.
            #  It could be safer to call model.zero_grad() if you have two or more optimizers for one model.
            loss, start_logits, end_logits = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
                token_type_ids=batch_token_type_ids,
                start_positions=batch_start_positions,
                end_positions=batch_end_positions
            )  # BertForQuestionAnswering uses CrossEntropyLoss by default, no need to calculate explicitly

            cumulative_train_loss_per_epoch += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            # clipping the norm of the gradients to 1.0 to help prevent the "exploding gradients" issues.
            optimizer.step()  # update model parameters
            lr_scheduler.step()  # update the learning rate

        average_training_loss_per_batch = cumulative_train_loss_per_epoch / len(train_dataloader)
        training_time = format_time(time() - t_i)
        logger.info(f"Epoch {epoch + 1} took {training_time} to train.")
        logger.info(f"Average training loss: {average_training_loss_per_batch}. \n Running validation.")
        if torch.cuda.is_available():
            logger.info(f"GPU memory usage: \n{gpu_memory_usage()}")

        t_i = time()
        model.eval()

        pred_start = torch.tensor([], dtype=torch.long, device=device)
        pred_end = torch.tensor([], dtype=torch.long, device=device)
        true_start = torch.tensor([], dtype=torch.long, device=device)
        true_end = torch.tensor([], dtype=torch.long, device=device)

        cumulative_eval_loss_per_epoch = 0

        for batch_num, batch in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
            logger.info(f"Running validation batch {batch_num + 1} of {len(valid_dataloader)}.")
            batch_input_ids, batch_token_type_ids, batch_attention_masks, batch_start_positions, batch_end_positions = \
                batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
            with torch.no_grad():
                loss, start_logits, end_logits = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_masks,
                    token_type_ids=batch_token_type_ids,
                    start_positions=batch_start_positions,
                    end_positions=batch_end_positions
                )
                cumulative_eval_loss_per_epoch += loss.item()

                pred_start_positions = torch.argmax(start_logits, dim=1)
                pred_end_positions = torch.argmax(end_logits, dim=1)

                pred_start = torch.cat((pred_start, pred_start_positions))
                pred_end = torch.cat((pred_end, pred_end_positions))
                true_start = torch.cat((true_start, batch_start_positions))
                true_end = torch.cat((true_end, batch_end_positions))
            if torch.cuda.is_available():
                logger.debug(f"GPU memory usage: \n{gpu_memory_usage()}")

        total_correct_start = int(sum(pred_start == true_start))
        total_correct_end = int(sum(pred_end == true_end))
        total_correct = total_correct_start + total_correct_end
        total_indices = len(true_start) + len(true_end)

        average_validation_accuracy_per_epoch = total_correct / total_indices
        average_validation_loss_per_batch = cumulative_eval_loss_per_epoch / len(valid_dataloader)
        valid_time = format_time(time() - t_i)
        logger.info(f"Epoch {epoch + 1} took {valid_time} to validate.")
        logger.info(f"Average validation loss: {average_validation_loss_per_batch}.")
        logger.info(f"Average validation accuracy (out of 1): {average_validation_accuracy_per_epoch}.")
        if torch.cuda.is_available():
            logger.info(f"GPU memory usage: \n{gpu_memory_usage()}")

        training_stats[f"epoch_{epoch + 1}"] = {
            "training_loss": average_training_loss_per_batch,
            "valid_loss": average_validation_loss_per_batch,
            "valid_accuracy": average_validation_accuracy_per_epoch,
            "training_time": training_time,
            "valid_time": valid_time
        }
        if save_model_path is not None:
            save_model_path = save_model_path.split(".")[0]  # removing extension if present
            torch.save(model.state_dict(), f"{save_model_path}_epoch_{epoch + 1}.pt")  # readd .pt extension
    if save_stats_dict_path is not None:
        with open(save_stats_dict_path, "w") as file:
            json.dump(training_stats, file)
    return model, training_stats
