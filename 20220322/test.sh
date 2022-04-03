#!/bin/sh

for var in 1 2 3 4  #範囲の書き方(Bash独自) => {0..4}
do
    python test_for_pfrl.py --steps 1000000 --seed $var --env Ant-v2 --agent SAC --gpu -1
done
