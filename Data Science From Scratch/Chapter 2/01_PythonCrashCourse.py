from __future__ import division
from functools import partial, reduce
import re
from collections import defaultdict, Counter
import random

# * Book: Data Science From Scratch by Joel Grus
# * page 40 (pdf file)

# * Title: Python Crash Curse
# Teorical Objective: Learn the basics of python for DS
# Practical Objective: None

# * What have I learned from this py file:
#   - Default Dicts in python
#   - Counter object from collections
#   - Sets
#   - Control Flow
#   - Thrutiness
#   - Sorting
#   - List comprehensions
#   - Generators and Iterators
#   - Randomness
#   - Regular Expressions
#   - Object Orientes Programming
#   - Functional Tools
#   - Enumerate
#   - zip and Unpacking Arguments
#   - args and kwargs

# * defaultdict()
word_counts = defaultdict(int)              # int() produces 0
document = ""
for word in document:
    word_counts[word] += 1

dd_list = defaultdict(list)                 # list() produces an empty list
dd_list[2].append(1)                        # now dd_list contains {2: [1]}

dd_dict = defaultdict(dict)                 # dict() produces an empty dict
dd_dict["Joel"]["City"] = "Seattle"         # { "Joel" : { "City" : Seattle"}}

dd_pair = defaultdict(lambda: [0, 0])
dd_pair[2][1] = 1                           # now dd_pair contains {2: [0,1]}

# * Counter()
c = Counter([0, 1, 2, 0])                   # c is (basically) { 0 : 2, 1 : 1, 2 : 1 }

# print the 10 most common words and their counts
for word, count in c.most_common(10):
    print(word, count)

# * Sets
# they represent collections of distinct elements
s = set()
s.add(1) # s is now { 1 }
s.add(2) # s is now { 1, 2 }
s.add(2) # s is still { 1, 2 }
x = len(s) # equals 2
y = 2 in s # equals True
z = 3 in s # equals False

# you can find distinct elements of a list with sets
item_list = [1, 2, 3, 1, 2, 3]
num_items = len(item_list) # 6
item_set = set(item_list) # {1, 2, 3}
num_distinct_items = len(item_set) # 3
distinct_item_list = list(item_set) # [1, 2, 3]

# * Control Flow
if 1 > 2:
    message = "if only 1 were greater than two…"
elif 1 > 3:
    message = "elif stands for 'else if'"
else:
    message = "when all else fails use else (if you want to)"

# ternary if-then-else in one line:
parity = "even" if x % 2 == 0 else "odd"

# while loops
x = 0
while x < 10:
    print(x, "is less than 10")
    x += 1

# for loops
for x in range(10):
    print(x, "is less than 10")

# continue and break
for x in range(10):
    if x == 3:
        continue                            # go immediately to the next iteration
    if x == 5:
        break                               # quit the loop entirely
    print(x)

# * Thruthiness
one_is_less_than_two = 1 < 2                # equals True
true_equals_false = True == False           # equals False

# None values
x = None
print(x == None)                                # prints True, but is not Pythonic
print(x is None)                                # prints True, and is Pythonic

# function all() returns True if all elements in it are True
all([True, 1, { 3 }])                           # True
all([True, 1, {}])                              # False, {} is falsy
any([True, 1, {}])                              # True, True is truthy
all([])                                         # True, no falsy elements in the list
any([])                                         # False, no truthy elements in the list

# * Sorting
x = [4,1,2,3]
y = sorted(x)                                   # is [1,2,3,4], x is unchanged
x.sort()                                        # now x is [1,2,3,4]

# sort the list by absolute value from largest to smallest
x = sorted([-4,1,-2,3], key=abs, reverse=True) # is [-4,3,-2,1]
# sort the words and counts from highest count to lowest
wc = sorted(word_counts.items(),
            # key=lambda (word, count): count,
            reverse=True)

# * List comprenhensions
even_numbers = [x for x in range(5) if x % 2 == 0]              # [0, 2, 4]
squares = [x * x for x in range(5)]                             # [0, 1, 4, 9, 16]
even_squares = [x * x for x in even_numbers]                    # [0, 4, 16]

# with dictionaries and sets
square_dict = { x : x * x for x in range(5) }                   # { 0:0, 1:1, 2:4, 3:9, 4:16 }
square_set = { x * x for x in [1, -1] }                         # { 1 }

# convention _ if you don't need value of the list
zeroes = [0 for _ in even_numbers]                              # has the same length as even_numbers

# multiple fors
pairs = [(x, y)
        for x in range(10)
        for y in range(10)]                                     # 100 pairs (0,0) (0,1) ... (9,8), (9,9)

increasing_pairs = [(x, y)                                      # only pairs with x < y,
                    for x in range(10)                          # range(lo, hi) equals
                    for y in range(x + 1, 10)]                  # [lo, lo + 1, ..., hi - 1]


# * Generators and Iterators
# A generator is something you can iterate over but whose values are produced
# only as needed (lazily).

# You can create one with functions and yield operator
def lazy_range(n):
    """A lazy version of range"""
    i = 0
    while i < n:
        yield i
        i += 1

# This for loop will consume the yielded values one at a time
for i in lazy_range(10):
    print(i)

# * Randomness
four_uniform_randoms = [random.random() for _ in range(4)]
# [0.8444218515250481, # random.random() produces numbers
# 0.7579544029403025, # uniformly between 0 and 1
# 0.420571580830845, # it's the random

# you can set an internal state with random.seed()
random.seed(10)                                                 # set the seed to 10
print(random.random())                                           # 0.57140259469
random.seed(10)                                                 # reset the seed to 10
print(random.random())                                           # 0.57140259469 again

# random numbers in a range
random.randrange(10)                           # choose randomly from range(10) = [0, 1, ..., 9]
random.randrange(3, 6)                         # choose randomly from range(3, 6) = [3, 4, 5]

# shuffle
up_to_ten = list(range(10))
random.shuffle(up_to_ten)
print(up_to_ten)
# [2, 5, 1, 9, 7, 3, 8, 6, 4, 0] (your results will probably be different)

# pick a random item from a list
my_best_friend = random.choice(["Alice", "Bob", "Charlie"]) # "Bob" for me

# random sample of a list without replacement
lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6) # [16, 36, 10, 6, 25, 9]

# random sambple of a list with replacement
four_with_replacement = [random.choice(range(10))
                        for _ in range(4)]
# [9, 4, 4, 2]

# * Regular Expressions
# this provide us a way of searching text
print(all([                                     # all of these are true, because
    not re.match("a", "cat"),                   # * 'cat' doesn't start with 'a'
    re.search("a", "cat"),                      # * 'cat' has an 'a' in it
    not re.search("c", "dog"),                  # * 'dog' doesn't have a 'c' in it
    3 == len(re.split("[ab]", "carbs")),        # * split on a or b to ['c','r','s']
    "R-D-" == re.sub("[0-9]", "-", "R2D2")      # * replace digits with dashes
    ])) # prints True

# * Object Oriented Programming
# by convention, we give classes PascalCase names
class SetClass:
    # these are the member functions
    # every one takes a first parameter "self" (another convention)
    # that refers to the particular Set object being used
    def __init__(self, values=None):
        """This is the constructor.
        It gets called when you create a new Set.
        You would use it like
        s1 = Set() # empty set
        s2 = Set([1,2,2,3]) # initialize with values"""
        self.dict = {} # each instance of Set has its own dict property
        # which is what we'll use to track memberships
        if values is not None:
            for value in values:
                self.add(value)

    def __repr__(self):
        """this is the string representation of a Set object
        if you type it at the Python prompt or pass it to str()"""
        return "Set: " + str(self.dict.keys())

    # we'll represent membership by being a key in self.dict with value True
    def add(self, value):
        self.dict[value] = True
    
    # value is in the Set if it's a key in the dictionary
    def contains(self, value):
        return value in self.dict

    def remove(self, value):
        del self.dict[value]

s = SetClass([1,2,3])
s.add(4)
print(s.contains(4)) # True
s.remove(3)
print(s.contains(3)) # False


# * Functional Tools
def exp(base, power):
    return base ** power

# you can do this
def two_to_the(power):
    return exp(2, power)

# different aproach
two_to_the = partial(exp, 2)                            # is now a function of one variable
print(two_to_the(3))                                    # 8

square_of = partial(exp, power=2)
print(square_of(3))                                     # 9

# an alternative to list comprehensions are the functions map, reduce and filter
def double(x):
    return 2 * x

xs = [1, 2, 3, 4]
twice_xs = [double(x) for x in xs]                      # [2, 4, 6, 8]
twice_xs = map(double, xs)                              # same as above
list_doubler = partial(map, double)                     # "function" that doubles a list
twice_xs = list_doubler(xs)                             # again [2, 4, 6, 8]

# map with functions of 2 variables
def multiply(x,y): return x * y
products = map(multiply, [1, 2], [4, 5])                # [1 * 4, 2 * 5] = [4, 10]

# similarly filter does the work for a list comprehension if
def is_even(x):
    """True if x is even, False if x is odd"""
    return x % 2 == 0

x_evens = [x for x in xs if is_even(x)]                 # [2, 4]
x_evens = filter(is_even, xs)                           # same as above
list_evener = partial(filter, is_even)                  # "function" that filters a list
x_evens = list_evener(xs)                               # again [2, 4]

# reduce
x_product = reduce(multiply, xs)                        # = 1 * 2 * 3 * 4 = 24
list_product = partial(reduce, multiply)                # "function" that reduces a list
x_product = list_product(xs)                            # again = 24

# * Enumerate
documents = ["Doc A", "Doc B", "Doc C"]
for i, document in enumerate(documents):
    print(i, document)

# if we only want the indexes
for i in range(len(documents)): print(i)         # not Pythonic
for i, _ in enumerate(documents): print(i)       # Pythonic

# * zip and Argument Unpacking
list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]
zip(list1, list2)                                       # is [('a', 1), ('b', 2), ('c', 3)]

# you can unzip lists using a trick
pairs = [('a', 1), ('b', 2), ('c', 3)]
letters, numbers = zip(*pairs)
# the asterix performs argument unpacking
zip(('a', 1), ('b', 2), ('c', 3))
# which returns [('a','b','c'), ('1','2','3')].

# you can use argument unpacking in any function:
def add(a, b): return a + b
add(1, 2)                                               # returns 3
#add([1, 2])                                             # TypeError!
add(*[1, 2])                                            # returns 3


# * args and kwargs
# let's say we want a higher order function that takes  as an input some function f
# and returns a new function that for any input returns twice the value of f
def doubler(f):
    def g(x):
        return 2 * f(x)
    return g

# this works in some cases
def f1(x):
    return x + 1
g = doubler(f1)
print(g(3))                              # 8 (== ( 3 + 1) * 2)
print(g(-1))                             # 0 (== (-1 + 1) * 2)

# however it breaks down with functions that take more than one argument
def f2(x, y):
    return x + y
g = doubler(f2)
#print(g(1, 2)) # TypeError: g() takes exactly 1 argument (2 given)

# we want to specify a function that takes arbitrary arguments
def magic(*args, **kwargs):
    print("unamed args: ", args)
    print("keyword args: ", kwargs)

magic(1, 2, key="word", key2="word2")
# prints
# unnamed args: (1, 2)
# keyword args: {'key2': 'word2', 'key': 'word'}

def other_way_magic(x, y, z):
    return x + y + z
x_y_list = [1, 2]
z_dict = { "z" : 3 }
print(other_way_magic(*x_y_list, **z_dict))                 # 6

def doubler_correct(f):
    """works no matter what kind of inputs f expects"""
    def g(*args, **kwargs):
        """whatever arguments g is supplied, pass them through to f"""
        return 2 * f(*args, **kwargs)
    return g
g = doubler_correct(f2)
print(g(1, 2))                                               # 6

