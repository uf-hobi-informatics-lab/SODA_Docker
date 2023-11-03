# -*- coding: utf-8 -*-

import random
import json
import pandas as pd
import argparse
import os, shutil

def main(file, curr_dir, name):
    #read input file and create needed directories
    input_file = pd.read_csv(file, sep='\t', header=0)
    if not os.path.exists(f'{curr_dir}'):
        os.mkdir(f'{curr_dir}')
    if not os.path.exists(f'{curr_dir}/raw_data'):
        os.mkdir(f'{curr_dir}/raw_data')

    for _, row in input_file.iterrows():

        with open(f'{curr_dir}/raw_data/{row["NOTE_ENCNTR_KEY"]}.txt', 'w') as fw:
            fw.write(str(row["note_text"]))
            #fw.write(row["TEXT"])

    #create config.yml in output dir
    with open(f'{curr_dir}/config.yml', 'w') as config:
        config.write(f'{name}:\n')
        config.write(f'  gpu_node: 0\n')
        config.write(f'  root_dir: {curr_dir}\n')
        config.write(f'  raw_data_dir: {curr_dir}/raw_data\n')
        config.write(f'  generate_bio: False\n')
        config.write(f'  encoded_text: True\n')
        config.write(f'  ner_model:\n')
        config.write(f'    type: bert\n')
        config.write(f'    path: ./models/ner/SDOH_bert_final\n')
        config.write(f'  rel_model:\n')
        config.write(f'    type: bert\n')
        config.write(f'    path: ./models/rel/bert')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="tsv file containing all note information")
    parser.add_argument("-d", "--directory", required=True, help="Output directory")
    parser.add_argument("-n", "--name", required=True, help="Process name")
    args = parser.parse_args()
    main(args.file, args.directory, args.name)
