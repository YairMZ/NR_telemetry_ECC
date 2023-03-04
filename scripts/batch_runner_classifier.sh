#!/bin/sh
clusters="$1"
max_thr="$2"
echo "Running with $clusters clusters"
echo "Running with 0.0$max_thr max threshold"
for i in $(seq "$max_thr")
do
  echo "Running with threshold=0.0$i"
  echo "python3 classifier_analysis.py --threshold 0.0$i --n_clusters $clusters"
  python3 classifier_analysis.py --threshold 0.0"$i" --n_clusters "$clusters"
done
