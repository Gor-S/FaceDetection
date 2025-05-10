import pyfiglet
from termcolor import colored

def print_banner_FF(text = "Welcome to\nFace Finder", color='cyan', font='slant'):
    banner = pyfiglet.figlet_format(text, font=font)
    print(colored(banner, color))

def print_banner_FE(text = "Welcome to\nNeuro Face", color='cyan', font='slant'):
    banner = pyfiglet.figlet_format(text, font=font)
    print(colored(banner, color))


