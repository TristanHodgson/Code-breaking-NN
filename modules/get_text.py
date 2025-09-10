from datasets import load_dataset
from itertools import islice
import re

from random import shuffle

from modules.caesar import char_to_num

from typing import List, Generator, Tuple, Optional
from tqdm.autonotebook import tqdm


def word_gen(stream: bool) -> Generator[str, None, None]:
    """
    A generator that yields words one by one from the manu/project_gutenberg dataset from Hugging Face

    Yields:
        str: A word from the dataset
    """
    try:
        dataset = load_dataset("manu/project_gutenberg", split="en", streaming=stream)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    for book in dataset:
        book_text = book.get("text", "").lower()
        book_text = re.sub(r"[^a-z\s]", " ", book_text)
        words = re.split(r"\s+", book_text)
        for word in words:
            if word:
                yield word


def fetch_chunks(num_blocks: int, words_per_block: int, stream: bool) -> List[str]:
    """
    Fetches chunks of text from the Project Gutenberg dataset
    Will chunk the whole dataset if num_blocks*words_per_block>len(dataset)

    Args:
        num_blocks (int): The number of text chunks to retrieve
        words_per_block (int): The number of words in each chunk

    Returns:
        List[str]: The final list of text chunks
    """
    word_stream = word_gen(stream)
    data = []
    for _ in range(num_blocks):
        block_words = list(islice(word_stream, words_per_block))

        if not block_words:
            print("End of dataset reached.")
            break
        data.append(" ".join(block_words))
    return data


def string2_num_list(string: str) -> list[int]:
    """
    Turns a string of spaces and letters into a list of numbers

    Args:
        char (str): The string to convert

    Returns:
        int: The numerical representation of the string. Returns 1-26 for 1-z, and 0 for a space
    """
    assert isinstance(string, str)
    nums = []
    for char in string:
        if char == " " or char.isalpha():
            nums.append(char_to_num(char)+1)
    return nums


def test_train_split(data: List, split: float) -> Tuple[List, List]:
    """
    Splits a list of data into random training and testing sets

    Args:
        data (List): The list of data to be split
        split (float): The proportion of the data to allocate to the training
                       set. Must be a value between 0.0 and 1.0

    Returns:
        Tuple[List, List]: A tuple containing two lists corresponding to training and testing data
    """
    assert isinstance(data, list)
    assert isinstance(split, (float, int))
    assert split >= 0 and split <= 1
    n = int(len(data)*split)
    shuffle(data)
    trainData, testData = data[:n], data[n:]
    return trainData, testData



def padding_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_value) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A collate function to pad sequences within a batch to the same length.

    This function is designed to be used with a `DataLoader`. It takes a list of
    (sequence, label) tuples, pads the sequences to the length of the longest
    sequence in the batch, and stacks the labels into a single tensor.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): A list of tuples from the
            Dataset, where each tuple contains a variable-length sequence tensor
            and a corresponding label tensor.
        pad_value int: The value that we want to use for padding

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - A tensor of padded sequences with shape (batch_size, max_seq_length).
            - A tensor of stacked labels with shape (batch_size,).
    """
    chunks, labels = zip(*batch)
    padded_chunks = pad_sequence(chunks, batch_first=True, padding_value=pad_value)
    stacked_labels = torch.stack(labels)
    return padded_chunks, stacked_labels