import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from modules.encryption import caesar
from modules.data_handling.get_text import *



def initialise(num_blocks: int,
               words_per_block: int,
               test_train_split_num: float = 0.8,
               stream: Optional[bool] = True) -> Tuple[List, List]:
    """
    Fetches a specified number of text chunks, encrypts each chunk using a caesar cipher, converts each character to a number, splits into random test and train datasets

    Args:
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
        chunk, key = caesar.encrypt(chunk)
        data.append([chunk, key])

    for i, (string, key) in enumerate(data):
        data[i] = [string2_num_list(string), key]
    return test_train_split(data, test_train_split_num)


class CipherDataset(Dataset):
    """
    Custom PyTorch Dataset for cipher text data.

    This dataset takes a list of integer sequences (chunks) and a corresponding
    list of integer labels. It prepares them for use in a PyTorch model by
    converting them to tensors.

    Args:
        chunks (List[List[int]]): A list of sequences, where each sequence is a
                                  list of integers representing encoded characters.
        labels (List[int]): A list of integer labels corresponding to each chunk.
    """
    def __init__(self, chunks: List[List[int]], labels: List[int]) -> None:
        self.chunks = chunks
        self.labels = labels

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset at the specified index.

        The method converts the integer sequence and its label into PyTorch tensors.
        Note that the label is adjusted by subtracting 1, assuming the original
        labels are 1-based and need to be converted to 0-based indices for
        purposes like loss calculation with `CrossEntropyLoss`.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the sequence
                                               tensor and the label tensor.
        """
        sequence = torch.tensor(self.chunks[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx] - 1, dtype=torch.long)
        return sequence, label

def pad_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    return padding_fn(batch, 27)

def data2loader(
    trainData: List[Tuple[List[int], int]],
    testData: List[Tuple[List[int], int]],
    BATCH_SIZE: int = 64
    ) -> Tuple[DataLoader, DataLoader]:
    """
    Converts raw training and testing data into PyTorch DataLoaders.

    This function automates the process of creating `CipherDataset` and `DataLoader`
    instances for both training and testing sets.

    Args:
        trainData (List[Tuple[List[int], int]]): A list of training samples, where
            each sample is a tuple containing a sequence (list of ints) and its
            label (int).
        testData (List[Tuple[List[int], int]]): A list of testing samples, structured
            identically to `trainData`.
        BATCH_SIZE (int, optional): The batch size for the DataLoaders. Defaults to 64.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training DataLoader
                                       and the testing DataLoader.
    """
    train_chunks = [datum[0] for datum in trainData]
    train_labels = [datum[1] for datum in trainData]
    test_chunks  = [datum[0] for datum in testData]
    test_labels  = [datum[1] for datum in testData]

    train_dataset = CipherDataset(train_chunks, train_labels)
    test_dataset = CipherDataset(test_chunks, test_labels)

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