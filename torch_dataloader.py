import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple


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
        """Initializes the CipherDataset."""
        self.chunks: List[List[int]] = chunks
        self.labels: List[int] = labels

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
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
    """
    A collate function to pad sequences within a batch to the same length.

    This function is designed to be used with a `DataLoader`. It takes a list of
    (sequence, label) tuples, pads the sequences to the length of the longest
    sequence in the batch, and stacks the labels into a single tensor. The padding
    value is set to 27.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): A list of tuples from the
            Dataset, where each tuple contains a variable-length sequence tensor
            and a corresponding label tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - A tensor of padded sequences with shape (batch_size, max_seq_length).
            - A tensor of stacked labels with shape (batch_size,).
    """
    chunks, labels = zip(*batch)
    padded_chunks = pad_sequence(chunks, batch_first=True, padding_value=27)
    stacked_labels = torch.stack(labels)
    return padded_chunks, stacked_labels

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
    train_chunks: List[List[int]] = [datum[0] for datum in trainData]
    train_labels: List[int] = [datum[1] for datum in trainData]
    test_chunks: List[List[int]] = [datum[0] for datum in testData]
    test_labels: List[int] = [datum[1] for datum in testData]

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