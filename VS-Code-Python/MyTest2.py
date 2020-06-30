import math
import sys
from os import rename

import requests

# Para este tipo de archivos .py donde recibimos un input()
# no lo podemos correr simplemente en CodeRunner, tenemos que dar
# click derecho y correr el .py en terminal
name = input("Your name? ")
print("Hello, ", name, "!")
