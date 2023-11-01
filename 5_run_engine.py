###
 # <p>Title:  </p>
 # <p>Create Date: 15:09:05 01/10/22</p>
 # <p>Copyright: College of Medicine </p>
 # <p>Organization: University of Florida</p>
 # @author Yonghui Wu
 # @version 1.0
 # <p>Description: </p>
 ##

from RuleEngine import RuleEngine
from TemporalExpression import *

def teExtract(te):
    max_len = 0
    final_te = []
    for expr in te:
        curr_len = len(expr.text)
        if curr_len > max_len:
            max_len = curr_len
            final_te = expr
    return final_te


if __name__ == "__main__":
    import os
    import re
    import sys
    import csv
    import shutil
    import argparse
    import pandas as pd

    forbidden_chars = ['?', '*', '[', ']', '@', '(', ')']
    rgx = re.compile('%s' % forbidden_chars)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="The file to read.")
    parser.add_argument("--re_map", type=str, default=None, help="An optional file to add.")
    parser.add_argument("--output_file", type=str, required=True, help="The desired output location.")
    args = parser.parse_args()

    # Check for metadata file
    metadata = False

    ''' 0. read dataframe from SDoH pipeline'''
    
    data = pd.read_csv(args.input_file)

    # SDoH_type,SDoH_value,SDoH_concept,SDoH_attributes,note_id
    data = data.rename(columns={'SDoH_type': 'SDoH_standard_category',
                                'SDoH_concept': 'SDoH_mention',
                                'note_id': 'NOTE_ENCNTR_KEY',
                                'SDoH_value': 'SDoH_raw_text'})
    if args.re_map is not None:
        rel_data = pd.read_csv(args.re_map)
        data = pd.concat([data, rel_data], ignore_index=True)
    data['SDoH_normalized'] = None
    if metadata:
        data = data[['NOTE_ENCNTR_KEY', 'patient_id', 'SDoH_standard_category',
                     'SDoH_mention','SDoH_raw_text','SDoH_normalized',
                     'SDoH_attributes','encounter_date']]
    else:
        data = data[['NOTE_ENCNTR_KEY', 'SDoH_standard_category', 'SDoH_mention', ## change this
                     'SDoH_raw_text','SDoH_normalized', 'SDoH_attributes']]
    #categories = data['SDoH_standard_category'].unique()
    #data['SDoH_normalized']=None
    #print(categories)
    #norm_data = pd.DataFrame()


    ''' 1. input text '''
 
    #print(current_data.head())
    norm_values = []
    my_engine= RuleEngine()
    for idx, text in enumerate(data['SDoH_raw_text']):
        #print(text, data['SDoH_standard_category'][idx])
        try:
            text = rgx.sub('', str(text).lower())
            #print(data['SDoH_standard_category'][idx])
            results=my_engine.extract(text, data['SDoH_standard_category'][idx])
            #print(results)
            if results != []:
                results = teExtract(results)
                #print(results.value)
                
                norm_values.append(results.value)
            else:
                norm_values.append('other')
        except:
            print(f"Skipping over {data['SDoH_standard_category'][idx]}. No rules found.")
            norm_values.append(None)
        
    data['SDoH_normalized']=norm_values
    #norm_data = pd.concat([norm_data, current_data])
    print(data)
    data['SDoH_normalized'] = data['SDoH_normalized'].fillna(value='other')
    data.to_csv(args.output_file, index=False, quoting=csv.QUOTE_ALL)