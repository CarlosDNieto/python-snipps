# video: https://www.youtube.com/watch?v=tJxcKyFMTGo
import datetime
import os

# nos permite interactuar con el sistema operativo

# imprime todos los métodos que hay en este modulo
# print(dir(os))

# imprime el directorio en el que nos encontramos
print("os.getcwd(): ", os.getcwd())

# cambiamos de directorio
path = "/Users/carlosnieto/Desktop/"
os.chdir(path)

print("Cambiamos de directorio con os.chdir(path): ", os.getcwd())

path = "/Users/carlosnieto/Desktop/Python-snippets"
os.chdir(path)

print("Regresamos al directorio original: ", os.getcwd())

# List of directories in the current directory
print(os.listdir())

# os.mkdir("nameofdir") para hacer un directorio
# os.mkdirs("nameofdir/nameofsubdir") para hacer directorios y subdirectorios en ese nuevo

# os.removedirs("nameofdir/nameofsubdir") para poder eliminar carpetas

# os.rename("nameoffileorfolder", "newname") para poder renombrar un archivo o un folder

# stats of a file
print("stats of emails.csv: ", os.stat("data/emails.csv"))
print("size of emails.csv: ", os.stat("data/emails.csv").st_size)

# imprime el tiempo de ultima modificacion
mod_time = os.stat("data/emails.csv").st_mtime
print(datetime.date.fromtimestamp(mod_time))

print()
# para poder 'caminar' sobre todos los directorios y archivos que están en un directorio dado
# podemos ocupar la siguiente funcion os.walk()
for namedir, directories, files in os.walk("/Users/carlosnieto/Desktop/Python-snippets"):
    print("Name: ", namedir)
    print("Directories: ", directories)
    print("Files: ", files)
    print()


# variables de entorno
home_path = os.environ.get("HOME")
print()
print(home_path)

# join strings to forma a path
data_path = '/Users/carlosnieto/Desktop/Python-snippets/data'
emails_file_name = 'emails.csv'
emails_path = os.path.join(data_path, emails_file_name)
print()
print(emails_path)

# solo el nombre del archivo o ultimo directorio
print()
print(os.path.basename(emails_path))

# el nombre del directorio
print()
print(os.path.dirname(emails_path))

# separar todos los elementos (basename y dirname) del path
print()
print(os.path.split(emails_path))

# checar si un path (archivo) existe
print()
print(os.path.exists(emails_path))

# hacer un directorio nuevo
tmp_path = os.path.join(os.getcwd(), "tmp")
# print(tmp_path)
if not os.path.exists(tmp_path):
    print("no existe tmp")
    os.mkdir(tmp_path)
else:
    print("ya existe tmp: ", tmp_path)


# agarra extensiones de archivos
print()
print(os.path.splitext(emails_path))
_, emails_ext = os.path.splitext(emails_path)
print("extencion del archivo de emails: ", emails_ext)


# todas los metodos de os.path
print()
print(dir(os.path))

# comentarios de Connor Cantrell en el video

# The OS module allows us to interact wiht the underlying operating system in several different ways.

# - Navigate the file system
# - Get file information
# - Look up and change the environment variables
# - Move files around
# - Many more

# To begin, import the os module. This is a built in module, no third party modules need to be installed.


# # Get current working directory
# os.getcwd()


# # Change directory, this requires a path to change to
# os.chdir(path)


# # List directory, you can pass a path, but by default it is in the current directory
# os.listdir()


# # Multiple options for creating directories
# mkdir()  # Use for making one directory
# makedirs(). # Use if you want to create multiple directories at once


# # Remove directories
# rmdir(file). # Recommended use case
# removedirs(file)  # Removes intermediate directories if specified


# # Rename a file or folder
# os.rename(‘test.txt’, ‘demo.txt’). # This renames text.txt to demo.txt


# # Look at info about files
# os.stat(test.txt)
# # Useful stat results: st_size (bytes), st_mtime (time stamp)


# # To see entire directory tree and files within
# # os.walk is a generator that yields a tuple of 3 values as it walks the directory tree

# for dirpath, dirnames, filenames in os.walk(routepath): 
#     print(‘Current Path:’, dirpath)
#     print(‘Directories:’, dirnames)
#     print(‘Files:’, filenames)
#     print()

# # This is useful for locating a file that you can’t remember where it was
# # If you had a web app, and you wanted to keep track of the file info within a certain directory structure, then you could to thru the os.walk method and go thru all files and folders and collect file information.


# # Access home directory location by grabbing home environment variable
# os.environ.get(‘HOME’). # Returns a path
# # To properly join two files together use os.path.join()
# file_path = os.path.join(os.environ.get(‘HOME’), ‘test.txt’)
# # the benefit of os.path.join, is it takes the guess work out of inserting a slash


# # os.path has other useful methods

# os.path.basename
# # This will grab filename of any path we are working on

# os.path.dirname(‘/tmp/test.txt’)
# # returns the directory /tmp

# os.path.split(‘/tmp/test.txt’)
# # returns both the directory and the file as a tuple

# os.path.exists(‘/tmp/test.txt’)
# # returns a boolean

# os.path.isdir(‘/tmp/test.txt’)
# # returns False

# os.path.isfile(‘/tmp/test.txt’)
# # returns True

# os.path.splitext(‘/tmp/test.txt’)
# # Splits file route of the path and the extension
# # returns (‘/tmp/test’, ‘.txt’)
# # This is alot easier than parsing out the extension. Splitting off and taking the first value is much better.
# # Very useful for file manipulation