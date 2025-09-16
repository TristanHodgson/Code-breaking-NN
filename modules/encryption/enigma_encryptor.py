from enigma.machine import EnigmaMachine
import string
import random

def rand_encrypt(text: str) -> str:
    rotors = ["I", "II", "III", "IV", "V"]
    rotors = random.sample(rotors, 3)
    rotors = ' '.join(rotors)

    reflectors = ["B", "C"]
    reflectors = random.choice(reflectors)

    rings = random.sample([str(i+1) for i in range(26)], 3)
    rings = ' '.join(rings)

    alphabet = list(string.ascii_uppercase)
    random.shuffle(alphabet)
    num_pairs = random.randint(5, 10)
    plug_pairs = [alphabet[i] + alphabet[i+1] for i in range(0, num_pairs * 2, 2)]
    plug_pairs_str = ' '.join(plug_pairs)
    
    message_key = ''.join(random.sample(string.ascii_uppercase, 3))

    machine = EnigmaMachine.from_key_sheet(
    rotors=rotors,
    reflector=reflectors,
    ring_settings=rings,
    plugboard_settings=plug_pairs_str
    )
    machine.set_display(message_key)
    return machine.process_text(text)