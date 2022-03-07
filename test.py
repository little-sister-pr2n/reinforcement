import random
from collections import namedtuple

Kind = namedtuple("Kind", 
    ("suit", "number")
)

card_list = list()
for i, mark in enumerate(["♠", "♥", "♦", "♣"]): # ♥♦♠♣
    for number in range(1,13+1):
        card_list.append(Kind(i, number))

print(sorted(random.sample(card_list, 5)))