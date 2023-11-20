#!/bin/bash
# Script to reproduce results

for ((i=0;i<5;i+=1))
do 
	python main-spik.py \
	--env "halfcheetah-random-v0" \
	--seed $i \
	--save_model
done
