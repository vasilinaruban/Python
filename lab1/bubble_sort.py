import random
from sys import argv

n = int(argv[1])
lst = [random.random() for i in range(n)]
for el in lst:
    print(el)

for i in range(n - 1):
    for j in range(n - i - 1):
        if lst[j] > lst[j + 1]:
            lst[j], lst[j + 1] = lst[j + 1], lst[j]

print("----------------")
for el in lst:
    print(el)