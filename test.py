import argparse
from time import sleep
parser = argparse.ArgumentParser()
parser.add_argument("--time", default=1, type=int)
args = parser.parse_args()
for t in range(args.time):
    sleep(1)
    print(t)