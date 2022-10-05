#!/bin/sh

while getopts c:n: flag
do
    case "${flag}" in
        c) config=${OPTARG};;
        n) nodes=${OPTARG};;
    esac
done

python ../run_ner.py --config $config --experiment $config --gpu_nodes $nodes
python ../run_relation.py --config $config --experiment $config --gpu_nodes $nodes
python ../adrd_output.py --config $config --experiment $config