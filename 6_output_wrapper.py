# -*- coding: utf-8 -*-

import random
import json
import pandas as pd
import argparse
import os, shutil, csv

#omop_cat = {"alcohol_use": 1234,
#            "so on": 50_47}

#omop_val = {"same as b": 4,
#             }

def main(file, structure, output):
    
    input_file = pd.read_csv(file)
    struct_file = pd.read_csv(structure, sep='\t')

    output_file = pd.merge(input_file, struct_file, how='right', on='NOTE_ENCNTR_KEY')
    #omop_cats = []
    #omop_vals = []
    #for _, row in output_file.iterrows():
    #    omop_cats.append(omop_cat[f'{row["SDoH_category"]}'])
    #    omop_vals.append(omop_val[f'{row["SDoH_normalized"]}'])
    #output_file["OMOP_category"] = omop_cats
    #output_file["OMOP_value"] = omop_vals
    output_file = output_file[['NOTE_ENCNTR_KEY', 'NOTE_KEY', 'CNTCT_NUM', 'SDoH_standard_category', 'SDoH_normalized']]#, 'OMOP_category', 'OMOP_value']]
    output_file.to_csv(f'{output}/extracted_sdoh_data.tsv', index=False, sep='\t', quoting=csv.QUOTE_ALL)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="csv output file containing all SDoH information")
    parser.add_argument("-s", "--structure", required=True, help="Input file to provide structure")
    parser.add_argument("-o", "--output", required=True, help="Output location/filename")
    args = parser.parse_args()
    main(args.file, args.structure, args.output)