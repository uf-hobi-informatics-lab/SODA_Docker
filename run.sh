#!/bin/bash

while getopts c:e:n:p: flag
do
    case "${flag}" in
        c) config=${OPTARG};;
        e) exper=${OPTARG};;
        n) nodes=${OPTARG};;
        p) prcssr=${OPTARG}
    esac
done

python run_ner.py --config $config --experiment $exper --gpu_nodes $nodes
python run_relation.py --config $config --experiment $exper --gpu_nodes $nodes
python adrd_output.py --config $config --experiment $exper --processors $prcssr