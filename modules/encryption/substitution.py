from typing import Tuple, List
from random import shuffle
from modules.helper import char_to_num

def encrypt(text: str, key: List[int]) -> Tuple[str, int]:
    """Encrypt a string using the substitution algorithm.

    The input text is converted to lowercase before encryption.
    Non-alphabetic characters (e.g., spaces, punctuation) are preserved in their original positions.

    Args:
        text (str): The string to be encrypted.
        key List[int]: A list of integers 0-25 inclusive that acts as the list of subsitutions to do, a->the letter associated with key[0]

    Returns:
        str: The substitution cipher encrypted string, in lowercase.
    
    Example:
        >>> encrypt("a:bcdefghijkl mnopqrstuvwxyz", [3, 11, 8, 5, 1, 20, 2, 15, 24, 25, 17, 7, 6, 18, 10, 13, 23, 4, 14, 0, 9, 19, 16, 12, 21, 22])
        d:lifbucpyzrh gsknxeoajtqmvw
    """
    text = text.lower()
    out = ""
    for char in text:
        if char.isalpha():
            position = char_to_num(char)
            position = (key[position]) + ord("a")
            out = out + chr(position)
        else:
            out = out + char
    return out


def rand_encrypt(text: str):
    """Encrypt a string using the substitution algorithm with a random key.

    Args:
        text (str): The string to be encrypted.

    Returns:
        str: The substitution cipher encrypted string, in lowercase.
    """
    key = [i for i in range(26)]
    shuffle(key)
    return encrypt(text, key)