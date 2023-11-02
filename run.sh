#!/bin/bash

nodes=""  # Initialize nodes as an empty string

while getopts f:d:n:g:p: flag
do
    case "${flag}" in
        f) file=${OPTARG};;
        d) directory=${OPTARG};;
        n) name=${OPTARG};;
        g) nodes=$(echo ${OPTARG} | sed 's/,/ /g');;  # Replace commas with spaces
        p) prcssr=${OPTARG};;
    esac
done
config=${directory}/config.yml

agg_loc=${directory}/output_csv/agg_output.csv
preproc_loc=${directory}/re_map.csv
norm_loc=${directory}/norm_output.csv

python 0_input_wrapper.py --file $file --directory $directory --name $name
python 1_run_ner.py --config $config --experiment $name --gpu_nodes $nodes
python 2_run_relation.py --config $config --experiment $name --gpu_nodes $nodes
python 3_aggregate_output.py --config $config --experiment $name --processors $prcssr
python 4_preprocess_for_normalization.py --directory $directory --processors $prcssr
python 5_run_engine.py --input_file $agg_loc --re_map $preproc_loc --output_file $norm_loc
python 6_output_wrapper.py --file $norm_loc --structure $file --output $directory
