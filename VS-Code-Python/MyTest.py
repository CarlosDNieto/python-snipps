import sys

import matplotlib.pyplot as plt
import pandas as pd
import requests

print(sys.version)
print(sys.executable)


def greet(who_to_greet):
    greeting = "Hello, {}!".format(who_to_greet)
    return greeting


print(greet("World"))
print(greet("Carlos"))

r = requests.get("https://github.com/CarlosDNieto")

print(r.status_code)
