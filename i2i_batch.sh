#!/bin/bash
for SEED in `seq 10000 1 10010` # seq [開始の数 [増分]] 終了の数
do
  echo SEED: $SEED
  python3 i2i.py --seed $SEED
done