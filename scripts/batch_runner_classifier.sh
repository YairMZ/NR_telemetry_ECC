#!/bin/sh
clusters="$1"
min_thr="$2"
inc_thr="$3"
max_thr="$4"

echo "Running with $clusters clusters"
for i in $(seq "$min_thr" "$inc_thr" "$max_thr");
do
  echo "Running with threshold=0.$i"
  echo "python3 classifier_analysis.py --threshold 0.$i --n_clusters $clusters"
  python classifier_analysis.py --threshold 0."$i" --n_clusters "$clusters"
done
