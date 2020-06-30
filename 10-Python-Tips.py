from getpass import getpass

# Tip 1:
# en vez de usar un if statement en varias lineas, usarlo en solo una
condition = True

# no hacer esto:
if condition:
    x = 1
else:
    x = 0

# hacer esto:
x = 1 if condition else 0

##########################################################################################

# Tip 2:
# usar underscores (_) para separar números grandes

num1 = 1_000_000
num2 = 100_000

total = num1 + num2

print(f'{total:,}')

##########################################################################################
print("")
# Tip 3:
# no abrir y cerrar archivos explícitamente si los vas a leer
# en vez de hacer eso mejor usa 'with'

with open('data/test.txt', 'r') as f:
    file_contents = f.read()

words = file_contents.split(" ")
word_count = len(words)
print(word_count)

##########################################################################################
print("")
# Tip 4:
# usar enumerate() y zip() para iterar sobre listas

names = ["Carlos", "Isaac", "Sebas", "Ramon"]

for index, name in enumerate(names, start=1):
    print(index, name)

names = ["Peter Parker", "Clark Kent", "Wade Wilson", "Bruce Wayne"]
heroes = ["Spiderman", "Superman", "Deadpool", "Batman"]
universes = ["Marvel", "DC", "Marvel", "DC"]

for index, name in enumerate(names):
    hero = heroes[index]
    print(f"{name} en realidad es {hero}")

for name, hero, universe in zip(names, heroes, universes):
    print(f"{name} en realidad es {hero} del universo {universe}")

##########################################################################################
print("")
# Tip 5:
# unpacking para una sola variable sin usar otras variables

a, _ = (1, 2)

print(a)

a, b, *c = (1, 2, 3, 4, 5)

print(a)
print(b)
print(c)

a, b, *c, d = (1, 2, 3, 4, 5)

print(a)
print(b)
print(c)
print(d)

##########################################################################################
print("")
# Tip 6:
# getting or setting attributes on a certain object


class Person:
    pass


person = Person()

person.last = "Nieto"
person.first = "Carlos"

print(person.first)
print(person.last)

person = Person()

first_key = "first"
first_value = "Carlos"

# asigna un atributo
setattr(person, first_key, first_value)

# print(person.first)

# obten un atributo
first = getattr(person, first_key)
# print(first)

# diccionario de atributos
person_info = {"first": "Carlos", "last": "Nieto"}
person = Person()

for key, value in person_info.items():
    setattr(person, key, value)

for key in person_info.keys():
    print(getattr(person, key))

##########################################################################################
print("")
# Tip 7:
# pedir contraseñas con la librería getpass

# correr en terminal
# username = input("Username: ")
# password = getpass("Password: ")

##########################################################################################
print("")
# Tip 8:
# 
