"""
Graphiti Commands
"""
from random import randint

def generate_random_pin_heights():
    """
    Generates a random pin height list
    """
    pin_list = []
    for _ in range(2400):
        pin_list.append((randint(1,4), 0))
    return pin_list

def update_display(pin_list):
    """
    Takes in a list of pin heights and blink times and converts into hexadecimal
    format readable by the Graphiti

    Args:
        pin_list: A list of length 2400 containing tuples of (pin_height, pin_blink)

    Returns:
        A hexadecimal value which the Graphiti can interpret
    """
    assert len(pin_list) == 2400

    hex_string = f'{27:#04x}'
    hex_string += f'{21:#04x}'

    for pin_height, pin_blink in pin_list:
        hex_string += f'{pin_height:#04x}'
        hex_string += f'{pin_blink:#04x}'
    
    return hex_string

def clear_display():
    """
    Creates hex command to clear the display and reset pin height to 0
    """
    hex_list = [27, 22, 3, 231]
    hex_string = ''
    for value in hex_list:
        hex_string += f'{value:#04x}'
    return hex_string

def enable_touch_mode():
    """
    Creates hex command enable touch
    """
    hex_list = [27, 65, 1, 190]
    hex_string = ''
    for value in hex_list:
        hex_string += f'{value:#04x}'
    return hex_string

def disable_touch_mode():
    """
    Creates hex command disabling touch
    """
    hex_list = [27, 65, 0, 191]
    hex_string = ''
    for value in hex_list:
        hex_string += f'{value:#04x}'
    return hex_string

