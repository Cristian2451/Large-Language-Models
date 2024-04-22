import torch
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Union
from transformers import PreTrainedTokenizerBase
import logging
from time import time
from .utils import format_time, set_hardware_acceleration


logger = logging.getLogger(__name__)


class DatasetEncoder:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, input_dataset: List[Dict]) -> None:
        expected_keys = ['context_text', 'question_text']
        assert all([key in dict_.keys() for key in expected_keys for dict_ in input_dataset]), \
            f"Each dictionary item in the input_dataset list must contain the following keys: {expected_keys}."
        self._tokenizer = tokenizer
        self._input_dataset = input_dataset

    def __len__(self):
        return len(self._input_dataset)

    def __getitem__(self, item):
        return self._input_dataset[item]

    @classmethod
    def from_dict_of_paragraphs(cls, tokenizer: PreTrainedTokenizerBase, input_dataset: Dict):
        assert 'data' in input_dataset.keys(), "SQuAD input dataset must have 'data' key."
        assert all([key in art.keys() for key in ['title', 'paragraphs'] for art in input_dataset['data']]), \
            "Input data don't match SQuAD structure. Keys 'title' and 'paragraphs' must be in each item in 'data' list."

        training_samples = cls(tokenizer, cls._create_training_samples_from_dict_of_paragraphs(input_dataset))
        return training_samples

    @staticmethod
    def _create_training_samples_from_dict_of_paragraphs(input_dict: Dict) -> List[Dict]:
        training_samples = []
        for article in input_dict['data']:
            for paragraph in article['paragraphs']:
                for qas in paragraph['qas']:  # each paragraph has multiple questions and answers associated
                    sample_dict = {
                        'answers': qas['answers'],
                        'context_text': paragraph['context'],
                        'qas_id': qas['id'],
                        'question_text': qas['question'],
                        'title': article['title']
                    }
                    training_samples.append(sample_dict)
        return training_samples

    def tokenize_and_encode(
            self,
            with_answers: bool,
            max_len: int,
            start_end_positions_as_tensors: bool = True,
            log_interval: Optional[int] = None,
            device_: Optional[str] = None  # if None, it automatically detects if a GPU is available, if not uses a CPU
    ) -> Tuple:
        if with_answers:
            assert all(['answers' in dict_.keys() for dict_ in self._input_dataset]), \
                "Not all questions provided contain an answer. If you do not intend to use ground truth answer " \
                "values for training or testing, please set with_answers=False ."
            return self._tokenize_and_encode_with_answer(max_len, start_end_positions_as_tensors, log_interval, device_)
        else:
            if not start_end_positions_as_tensors:
                logger.warning("Setting start_end_positions_as_tensors=False has no effect when with_answers=False.")
            if log_interval is not None:
                logger.warning("Setting log_interval has no effect when with_answers=False")
            return self._tokenize_and_encode_without_answer(max_len, device_)

    def _tokenize_and_encode_with_answer(
            self,
            max_len: int,
            start_end_positions_as_tensors: bool = True,
            log_interval: Optional[int] = None,
            device_: Optional[str] = None  # if None, it automatically detects if a GPU is available, if not uses a CPU
    ) -> Tuple[Tensor, Tensor, Tensor, Union[List[List[Tensor]], Tensor], Union[List[List[Tensor]], Tensor], int]:

        device = set_hardware_acceleration(default=device_)

        dropped_samples = 0
        all_encoded_dicts = []
        all_q_start_positions = []
        all_q_end_positions = []

        t_i = time()  # initial time
        for i, sample in enumerate(self._input_dataset):
            if start_end_positions_as_tensors and len(sample['answers']) != 1:
                raise IndexError(
                    f"In order to return torch tensors for training, each question must have only one possible "
                    f"answers. If tokenizing questions with multiple valid answers for testing, please set "
                    f"start_end_positions_as_tensors=False."
                )
            if log_interval is not None and i % log_interval == 0 and i != 0:
                logger.info(
                    f"Encoding sample {i} of {len(self._input_dataset)}. Elapsed: {format_time(time() - t_i)}. "
                    f"Remaining: {format_time((time() - t_i) / i * (len(self._input_dataset) - i))}."
                )
            possible_starts = []
            possible_ends = []
            # in dev sets with more than one possible answer, it records if some but not all valid answers are truncated
            for possible_answer in sample['answers']:
                answer_tokens = self._tokenizer.tokenize(possible_answer['text'])
                answer_replacement = " ".join(["[MASK]"] * len(answer_tokens))
                start_position_character = possible_answer['answer_start']
                end_position_character = possible_answer['answer_start'] + len(possible_answer['text'])
                context_with_replacement = sample['context_text'][:start_position_character] + answer_replacement + \
                    sample['context_text'][end_position_character:]
                encoded_dict = self._tokenizer.encode_plus(
                    sample['question_text'],
                    context_with_replacement,
                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]' tokens
                    max_length=max_len,
                    padding='max_length',  # Pad or truncates sentences to `max_length`
                    truncation=True,
                    return_attention_mask=True,  # Construct attention masks.
                    return_tensors='pt',  # Return pytorch tensors.
                ).to(device)

                is_mask_token = encoded_dict['input_ids'].squeeze() == self._tokenizer.mask_token_id
                mask_token_indices = is_mask_token.nonzero(as_tuple=False)
                if len(mask_token_indices) != len(answer_tokens):
                    continue  # ignore cases where start or end of answer exceed max_len and have been truncated
                answer_start_index, answer_end_index = mask_token_indices[0], mask_token_indices[-1]
                possible_starts.append(answer_start_index)
                possible_ends.append(answer_end_index)
                answer_token_ids = self._tokenizer.encode(
                    possible_answer['text'],
                    add_special_tokens=False,
                    return_tensors='pt'
                ).to(device)
            if len(sample['answers']) != len(possible_starts) or len(sample['answers']) != len(possible_ends):
                dropped_samples += 1  # we drop sample due to answer being truncated
                continue
            encoded_dict['input_ids'][0, answer_start_index:answer_end_index + 1] = answer_token_ids
            # Finally, replace the "[MASK]" tokens with the actual answer tokens
            all_encoded_dicts.append(encoded_dict)
            all_q_start_positions.append(possible_starts)
            all_q_end_positions.append(possible_ends)

        assert len(all_encoded_dicts) == len(self._input_dataset) - dropped_samples, "Lengths check failed!"
        input_ids = torch.cat([encoded_dict['input_ids'] for encoded_dict in all_encoded_dicts], dim=0)
        token_type_ids = torch.cat([encoded_dict['token_type_ids'] for encoded_dict in all_encoded_dicts], dim=0)
        attention_masks = torch.cat([encoded_dict['attention_mask'] for encoded_dict in all_encoded_dicts], dim=0)
        if start_end_positions_as_tensors:
            all_q_start_positions = torch.tensor(all_q_start_positions).squeeze().to(device)
            all_q_end_positions = torch.tensor(all_q_end_positions).squeeze().to(device)
        if dropped_samples > 0:
            logger.warning(
                f"Dropped {dropped_samples} question+context pair samples from the dataset because the start or end "
                f"token of the answer was at an unreachable position exceeding the max_len ({max_len})."
            )
        return input_ids, token_type_ids, attention_masks, all_q_start_positions, all_q_end_positions, dropped_samples

    def _tokenize_and_encode_without_answer(
            self,
            max_len: int,
            device_: Optional[str] = None  # if None, it automatically detects if a GPU is available, if not uses a CPU
    ) -> Tuple[Tensor, Tensor, Tensor]:

        device = set_hardware_acceleration(default=device_)

        all_encoded_dicts = []
        for i, sample in enumerate(self._input_dataset):
            encoded_dict = self._tokenizer.encode_plus(
                sample['question_text'],
                sample['context_text'],
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            ).to(device)
            all_encoded_dicts.append(encoded_dict)
        input_ids = torch.cat([encoded_dict['input_ids'] for encoded_dict in all_encoded_dicts], dim=0)
        token_type_ids = torch.cat([encoded_dict['token_type_ids'] for encoded_dict in all_encoded_dicts], dim=0)
        attention_masks = torch.cat([encoded_dict['attention_mask'] for encoded_dict in all_encoded_dicts], dim=0)
        return input_ids, token_type_ids, attention_masks
