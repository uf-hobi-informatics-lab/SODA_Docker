#save all functions in norm
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import shutil
import pickle as pkl
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
def pkl_dump(data, file):
    with open(file, "wb") as fw:
        pkl.dump(data, fw)

        
        
def pkl_load(file):
    with open(file, "rb") as fr:
        data = pkl.load(fr)
    return data


def helper(fids):
    sdf = pd.DataFrame(columns=['note_id', 're_type','SDoH_standard_category_1', 'SDoH_mention_1', 'SDoH_standard_category_2', 'SDoH_mention_2'])
    for idx, fid in enumerate(fids):
        note_id = fid.stem
        ner = read_brat_2(fid)
        #print(ner)
        concepts=read_brat_re(fid,ner)
        #print(concepts)
        if (idx+1) % 10000 == 0:
            pid = multiprocessing.current_process()
        for concept in concepts:
            data = {'note_id': str(note_id), 're_type': str(concept[1]),'SDoH_standard_category_1': str(concept[2]), 'SDoH_mention_1': str(concept[3]).replace("\n", ""), ## And here, add context as relation ArgX:ArgY 
                    'SDoH_standard_category_2': str(concept[4]),'SDoH_mention_2': str(concept[5]).replace("\n", "")}
            mdf = pd.DataFrame([data])
            sdf = pd.concat([sdf, mdf], ignore_index=True)
    return sdf





def parse_brat_ner(brat_data):
	info = brat_data.split("\t")
	idx = info[0]
	text = info[2]
	tse = info[1]
	if ";" in tse:
		ii = tse.split(" ")
		return [idx, ii[0], " ".join(ii[1:-1]), ii[-1], text]
	else:
		tag, s, e = tse.split(" ")
		return [idx, tag, int(s), int(e), text]
    
def parse_brat_re(brat_data,ner):  ##From here, pass on the concept
    info = brat_data.split("\t")
    idx = info[0]
    text = info[1]
    text_info=text.split(' ')
    label=text_info[0]
    arg1=text_info[1].split(':')[1]
    arg2=text_info[2].split(':')[1]
    re1=ner[arg1]
    re2=ner[arg2]
    return [idx, label,re1[0], re1[1],re2[0],re2[1]]



def read_brat_2(file_name):
	ners = dict()
	with open(file_name, "r") as f:
		cont = f.read().strip()
	if not cont:
		return ners
	for each in cont.split("\n"):
		if each.startswith("T"):
			ners.update({parse_brat_ner(each)[0]:[parse_brat_ner(each)[1],parse_brat_ner(each)[4]]})
		else:
			continue
			# raise RuntimeError('invalid brat data: {}'.format(each))
	return ners

def read_brat_re(file_name,ner): ## Here, establish the concepts as context
    reres = []
    with open(file_name, "r") as f:
        cont = f.read().strip()
    if not cont:
        return reres
    for each in cont.split("\n"):
        if each.startswith("R"):
            #print(each)
            reres.append(parse_brat_re(each,ner))
        else:
            continue
        # raise RuntimeError('invalid brat data: {}'.format(each))
    return reres
def main(directory, processors):
    re_list = []
    data_dir = f'{directory}/brat_re'
    local_re_list = list(Path(data_dir).glob("*.ann"))
    re_list.append(local_re_list)

    re_list =  [item for sublist in re_list for item in sublist]
    print(f'{len(re_list)} notes')

    N = int(processors)
    inputs = np.array_split(re_list, N)
    df = pd.DataFrame(columns=['note_id', 're_type','SDoH_standard_category_1', 'SDoH_mention_1', 'SDoH_standard_category_2', 'SDoH_mention_2'])
    with ProcessPoolExecutor(max_workers=N) as exe:
        for each in exe.map(helper, inputs):
            df = pd.concat([df, each])

    df.rename(columns={'SDoH_type_1': 'SDoH_standard_category_1', 'SDoH_type_2': 'SDoH_standard_category_2'}, inplace=True)
    new_cat = []
    new_raw_val = []
    new_mention = []

    to_lb=[ 'Tobacco_use','Smoking_freq_ppd',      'Smoking_freq_qy',      'Smoking_freq_sy',
            'Smoking_type', 'Smoking_freq_py','Smoking_freq_other',]

    dr_lb=[ 'Drug_use','Drug_type',   'Drug_freq',
            'Drug_other',]

    al_lb=[ 'Alcohol_use', 'Alcohol_freq',  'Alcohol_type',
            'Alcohol_other', ]

    for idc, row in df.iterrows():
        if ((row['SDoH_standard_category_1']=='Substance_use_status') or (row['SDoH_standard_category_1']=='Sdoh_status') or (row['SDoH_standard_category_1']=='Sdoh_freq')):
            if row['SDoH_standard_category_2'] in to_lb:
                new_cat.append('Tobacco_use')
            elif row['SDoH_standard_category_2'] in dr_lb:
                new_cat.append('Drug_use')
            elif row['SDoH_standard_category_2'] in al_lb:
                new_cat.append('Alcohol_use')
            else:    
                new_cat.append(row['SDoH_standard_category_2'])
            new_raw_val.append(row['SDoH_mention_1'])
            new_mention.append(row['SDoH_mention_2'])
        elif ((row['SDoH_standard_category_2']=='Substance_use_status') or (row['SDoH_standard_category_2']=='Sdoh_status') or (row['SDoH_standard_category_2']=='Sdoh_freq')):
            if row['SDoH_standard_category_1'] in to_lb:
                new_cat.append('Tobacco_use')
            elif row['SDoH_standard_category_1'] in dr_lb:
                new_cat.append('Drug_use')
            elif row['SDoH_standard_category_1'] in al_lb:
                new_cat.append('Alcohol_use')
            else:    
                new_cat.append(row['SDoH_standard_category_1'])
            new_raw_val.append(row['SDoH_mention_2'])
            new_mention.append(row['SDoH_mention_1'])
        else:
            new_cat.append(None)
            new_raw_val.append(None)
            new_mention.append(None)
            
    df['SDoH_standard_category'] = new_cat
    df['SDoH_raw_text'] = new_raw_val
    df['SDoH_mention'] = new_mention
    df = df[['note_id', 'SDoH_standard_category', 'SDoH_mention','SDoH_raw_text']]

    df.to_csv(f'{directory}/re_map.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True, help="Output directory")
    parser.add_argument("-p", "--processors", required=True, help="Processor number")
    args = parser.parse_args()
    main(args.directory, args.processors)