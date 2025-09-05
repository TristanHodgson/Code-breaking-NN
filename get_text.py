from datasets import load_dataset
from itertools import islice
import re

from random import shuffle

from typing import List, Generator, Tuple
from tqdm.autonotebook import tqdm


def word_gen() -> Generator[str, None, None]:
    """
    A generator that yields words one by one from the manu/project_gutenberg dataset from Hugging Face

    Yields:
        str: A word from the dataset
    """
    try:
        dataset = load_dataset("manu/project_gutenberg", split="en")
        # dataset = load_dataset("manu/project_gutenberg", split="en", streaming=True)
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


def fetch_chunks(num_blocks: int, words_per_block: int) -> List[str]:
    """
    Fetches chunks of text from the Project Gutenberg dataset
    Will chunk the whole dataset if num_blocks*words_per_block>len(dataset)

    Args:
        num_blocks (int): The number of text chunks to retrieve
        words_per_block (int): The number of words in each chunk

    Returns:
        List[str]: The final list of text chunks
    """
    word_stream = word_gen()
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
               test_train_split_num: float = 0.8) -> Tuple[List, List]:
    """
    Fetches a specified number of text chunks, encrypts each chunk using a provided encryption algorithm, shuffles the resulting data, and then splits it into a training set and a testing set

    Args:
        encryptionAlgorithm (function): A function that takes a plaintext string and returns a tuple containing the corresponding ciphertext (str) and the random encryption key (int)
        num_blocks (int): The total number of chunks of text to fetch
        words_per_block (int): The number of words in each chunk.
        test_train_split_num (float, optional): The proportion of the data to allocate to the training set. Defaults to 0.8

    Returns:
        tuple:
            - The training dataset: A list of [ciphertext, key] pairs
            - The testing dataset: A list of [ciphertext, key] pairs
    """
    textChucks = fetch_chunks(num_blocks, words_per_block)

    data = []
    for chunk in tqdm(textChucks):
        chunk, key = encryptionAlgorithm(chunk)
        data.append([chunk, key])

    return test_train_split(data, test_train_split_num)


def test_train_split(data: List, split: float) -> Tuple(List, List):
    """
    Splits a list of data into random training and testing sets

    Args:
        data (List): The list of data to be split
        split (float): The proportion of the data to allocate to the training
                       set. Must be a value between 0.0 and 1.0

    Returns:
        Tuple[List[T], List[T]]: A tuple containing two lists:
                                 - The first list is the training data.
                                 - The second list is the testing data.
    """
    assert isinstance(data, list)
    assert isinstance(split, (float, int))
    assert split >= 0 and split <= 1
    n = int(len(data)*split)
    shuffle(data)
    trainData, testData = data[:n], data[n:]
    return trainData, testData


if __name__ == "__main__":
    number_of_blocks = 5
    words_per_block = 200

    blocks = fetch_chunks(number_of_blocks, words_per_block)
    if blocks:
        for i, block in enumerate(blocks):
            print(f'{"#"*51}\n{"#"*18} Block {i:03d} {"#"*18}\n{"#"*51}{block}\n')
    else:
        print("Could not generate text blocks.")
