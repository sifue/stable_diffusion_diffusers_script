#!/bin/bash
for SEED in `seq 20000 1 21000` # seq [開始の数 [増分]] 終了の数
do
  echo SEED: $SEED
  python3 t2i.py --seed $SEED
done