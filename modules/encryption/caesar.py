from random import randint
from typing import Optional, Tuple
from modules.helper import char_to_num




def encrypt(text: str, key: Optional[int] = None) -> Tuple[str, int]:
    """
    Encrypt a string using the Caesar cipher algorithm.
    The input text is converted to lowercase before encryption.
    Non-alphabetic characters (e.g., spaces, punctuation) are preserved in their original positions.

    Args:
        text (str): The string to be encrypted.
        key Optional[int]: The integer offset for the character shift in the forward direction, if None provided we use a random key

    Returns:
        str: The Caesar cipher encrypted string, in lowercase.

    Example:
        >>> encrypt("Hello World!", key=3)
        'khoor zruog!'
    """
    text = text.lower()
    out = ""
    if not key:
        key = randint(1, 25)
    for char in text:
        if char.isalpha():
            position = char_to_num(char)
            position = (position + key) % 26 + ord("a")
            out = out + chr(position)
        else:
            out = out + char
    return out, key


def decrypt(text: str, key: int) -> str:
    """
    Decrypt a string encrypted with the Caesar cipher..

    Args:
        text (str): The encrypted string to be decrypted.
        key (int): The integer key that was used for encryption.

    Returns:
        str: The decrypted string.

    Example:
        >>> decrypt("khoor zruog!", 3)
        'hello world!'
    """
    return encrypt(text, key=-key)[0]


def rand_encrypt(text:str) -> str:
    return encrypt(text)[0]

if __name__ == "__main__":
    key = 2
    enc = encrypt("Hello World", key)[0]
    dec = decrypt(enc, key)
    print(f"{dec} -> {enc}")
