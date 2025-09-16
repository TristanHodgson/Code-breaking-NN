def char_to_num(char: str) -> int:
    """
    Converts a single alphabet character or a space into a numerical representation.

    Args:
        char (str): The single alphabetical character or a space

    Returns:
        int: The numerical representation of the character. Returns 0-25 for 1-z, and -1 for a space

    Raises:
        AssertionError: If the input string `char` is not a single character,
                        or if it is not an alphabet character or a space
    """
    assert len(char) == 1, "Error, length of string passed to char_to_num() should be 1"
    if char.isalpha():
        return ord(char) - ord("a")
    elif char == " ":
        return -1
    else:
        assert False, "Error, char_to_num() only accepts alphabet or space characters"


