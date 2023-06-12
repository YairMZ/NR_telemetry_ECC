#!/bin/sh
clusters="$1"
multiply_data="$2"


echo "Running with $clusters clusters"
echo "Running with $multiply_data multiply_data"
for context_length in $(seq 1 1 8);
do
  echo "Running with context_length=$context_length"
  echo "python3 simulated_data_classifying_dude_decoder.py --minflip 0.04 --maxflip 0.07 --nflips 5 --multiply_data $multiply_data  --classifier_train 100 --n_clusters $clusters --context_length $context_length"
  python simulated_data_classifying_dude_decoder.py --minflip 0.04 --maxflip 0.07 --nflips 5 --multiply_data "$multiply_data"  --classifier_train 100 --n_clusters "$clusters" --context_length "$context_length"
done
