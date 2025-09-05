from datasets import load_dataset
from itertools import islice
import re

from random import shuffle

from ceaser import char_to_num

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


def initialise(encryptionAlgorithm,
               num_blocks: int,
               words_per_block: int,
               test_train_split_num: float = 0.8,
               stream: Optional[bool] = True) -> Tuple[List, List]:
    """
    Fetches a specified number of text chunks, encrypts each chunk using a provided encryption algorithm, converts each character to a number, splits into random test and train datasets

    Args:
        encryptionAlgorithm (function): A function that takes a plaintext string and returns a tuple containing the corresponding ciphertext (str) and the random encryption key (int)
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
        chunk, key = encryptionAlgorithm(chunk)
        data.append([chunk, key])

    for i, (string, key) in enumerate(data):
        data[i] = [string2_num_list(string), key]
    return test_train_split(data, test_train_split_num)


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