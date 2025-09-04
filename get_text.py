import re
from datasets import load_dataset
from typing import List, Generator
from itertools import islice

def word_gen() -> Generator[str, None, None]:
    """
    A generator that yields words one by one from the manu/project_gutenberg dataset from Hugging Face

    Yields:
        str: A word from the dataset
    """
    try:
        dataset = load_dataset("manu/project_gutenberg", split="en", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    for book in dataset:
        book_text = book.get("text", "")
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


if __name__ == "__main__":
    number_of_blocks = 5
    words_per_block = 200
    
    blocks = fetch_chunks(number_of_blocks, words_per_block)
    if blocks:
        for i, block in enumerate(blocks):
            print(f"\n--- Block {i+1} (approx. {len(block.split())} words) ---")
            print(block)
    else:
        print("Could not generate text blocks.")
