import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from modules.get_text import *


def initialise(encryptionAlgorithm,
               num_blocks: int,
               words_per_block: int,
               test_train_split_num: float = 0.8,
               stream: Optional[bool] = True) -> Tuple[List, List]:
    """
    Fetches a specified number of text chunks, encrypts each chunk using a provided encryption algorithm, converts each character to a number, splits into random test and train datasets

    Args:
        encryptionAlgorithm (function): A function that takes a plaintext string and returns the corresponding encrypted text
        num_blocks (int): The total number of chunks of text to fetch
        words_per_block (int): The number of words in each chunk.
        test_train_split_num (float, optional): The proportion of the data to allocate to the training set. Defaults to 0.8

    Returns:
        tuple:
            - The training dataset: A list of [numeric ciphertext, key] pairs
            - The testing dataset: A list of [numeric ciphertext, key] pairs
    """
    textChucks = fetch_chunks(num_blocks, words_per_block, stream)

    data = []
    for chunk in tqdm(textChucks):
        enc = encryptionAlgorithm(chunk)
        data.append([chunk, enc])

    for i, (string, encString) in enumerate(data):
        string = [29] + string2_num_list(string) + [28]
        encString = [29] + string2_num_list(encString) + [28]
        data[i] = [string, encString]
    return test_train_split(data, test_train_split_num)


class Seq2SeqDataset(Dataset):
    def __init__(self, data_pairs: List[List[List[int]]]):
        self.data_pairs = data_pairs

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        source_seq, target_seq = self.data_pairs[idx]
        source_tensor = torch.tensor(source_seq, dtype=torch.long)
        target_tensor = torch.tensor(target_seq, dtype=torch.long)
        return source_tensor, target_tensor


def pad_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    return padding_fn(batch, 27, True)


def data2loader(
    trainData: List[Tuple[List[int], int]],
    testData: List[Tuple[List[int], int]],
    BATCH_SIZE: int = 64
    ) -> Tuple[DataLoader, DataLoader]:
    
    train_dataset = Seq2SeqDataset(trainData)
    test_dataset = Seq2SeqDataset(testData)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Not necessary to be true as we already shuffled the next to ensure a proper test/train split
        collate_fn=pad_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_fn
    )
    return train_loader, test_loader