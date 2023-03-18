#!/bin/sh
n_runs="$1"
p_min="$2"
p_max="$3"
inc_p="$4"

echo "Running clsutering_analysis with $n_runs runs"
for i in $(seq "$p_min" "$inc_p" "$p_max");
do
  echo "Running with p=0.$i"
  echo "python3 clustering_analysis.py --p 0.$i --n_runs $n_runs"
  python clustering_analysis.py --p "0.$i" --n_runs "$n_runs"
done
