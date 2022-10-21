#!/bin/bash

while getopts c:n: flag
do
    case "${flag}" in
        c) config=${OPTARG};;
        x) exper=${OPTARG};;
        n) nodes=${OPTARG};;
    esac
done

python run_ner.py --config $config --exper $experiment --gpu_nodes $nodes
python run_relation.py --config $config --exper $experiment --gpu_nodes $nodes