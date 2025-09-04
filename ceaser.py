def encrypt(text: str, key: int) -> str:
    """
    Encrypt a string using the Caesar cipher algorithm.
    The input text is converted to lowercase before encryption.
    Non-alphabetic characters (e.g., spaces, punctuation) are preserved in their original positions.

    Args:
        text (str): The string to be encrypted.
        key (int): The integer offset for the character shift in the forward direction.

    Returns:
        str: The Caesar cipher encrypted string, in lowercase.

    Example:
        >>> encrypt("Hello World!", 3)
        'khoor zruog!'
    """
    text = text.lower()
    out = ""
    for char in text:
        if char.isalpha():
            position = ord(char) - ord("a")
            position = (position + key) % 26 + ord("a")
            out = out + chr(position)
        else:
            out = out + char
    return out

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
    return encrypt(text, -key)


if __name__ == "__main__":
    key = 2
    enc = encrypt("Hello World", key)
    dec = decrypt(enc, key)
    print(f"{dec} -> {enc}")