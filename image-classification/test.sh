for epoch in {10..30} 
do 
	python test_score.py --image-shape 3,112,112 --test-rec wxb-pulp-test-112.rec --pretrained-model save/lightpulp --load-epoch $epoch
        echo $epoch
done
