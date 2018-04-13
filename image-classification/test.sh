#!/bin/bash

for thresh in {2..10} 
do 
        epoch=30
        a=0.01
        e=$(echo "$thresh * $a" | bc)
        echo $e
	python test_score.py --thrshold $e --image-shape 3,224,224 --test-rec wxb-pulp-test.rec  --pretrained-model save/lightpulp224_A --load-epoch $epoch
        echo $epoch
done
