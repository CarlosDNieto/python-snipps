from collections import OrderedDict, Counter

class OrderedCounter(OrderedDict, Counter):
    pass

n = 4
inputs = ['bcdef', 'abcdefg', 'bcde', 'bcdef']

d = OrderedCounter()
for word in inputs:
    d[word] += 1
print(d)
print(len(d))
print(*d.values())
