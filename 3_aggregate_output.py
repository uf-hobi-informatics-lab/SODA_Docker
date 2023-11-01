import argparse
import copy
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import yaml

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def get_report_meta(brat):
    re_format = r'(IRB\d+)_(NOTE_\d+)_(\w+)_(IRB\d+)_(PAT_\d+)'
    irb_id, note_id, note_ver, _, pat_id = re.findall(re_format,brat.stem)[0]
    return irb_id, note_id, note_ver, pat_id

def gen_adrd_output_df(df):

    df['SDoH_type'] = np.nan
    df['SDoH_concept'] = np.nan
    df['SDoH_value'] = np.nan
    
    # group multiple SDoH_value
    df = pd.concat([df.loc[~df['child_concept_cat'].isin(['Substance_use_status', 'Sdoh_status'])],\
                    df.loc[df['child_concept_cat'].isin(['Substance_use_status', 'Sdoh_status'])].groupby(
                        ['id', 'concept_cat', 'concept_value', 'child_concept_cat', 'relation', 'context', 'child_context'], dropna=False)['child_concept_value'].apply('|'.join).reset_index()])
    
    # No child
    df.loc[df['child_concept_cat'].isnull(),'SDoH_type'] = df.loc[df['child_concept_cat'].isnull()]['concept_cat']
    df.loc[df['child_concept_cat'].isnull(),'SDoH_value'] = df.loc[df['child_concept_cat'].isnull()]['concept_value']
    
    # label == substabce_use_status
    # if df.loc[(df['concept_cat']=='Substance_use_status') & ~df['concept_value'].str.contains('smok|drug',case=False) & ~df['child_concept_cat'].str.contains('smok|drug', na=False,case=False)].shape[0]:
    #     pprint(df)
    for k,v in zip(['smok','drug','alcoh'],['Tobacco_use', 'Drug_use', 'Alcohol_use']):
        df.loc[(df['concept_cat']=='Substance_use_status') & (df['concept_value'].str.contains(k,case=False) | df['child_concept_cat'].str.contains(k, na=False,case=False)),'SDoH_type'] = v
    df.loc[df['concept_cat']=='Substance_use_status','SDoH_value'] = df.loc[df['concept_cat']=='Substance_use_status']['concept_value']
    
    # child_label in [substance_use_status, sdoh_status]
    df.loc[~df['concept_cat'].isin(['Substance_use_status', 'Sdoh_status']) & ~df['relation'].isnull(),'SDoH_type'] = \
        df.loc[~df['concept_cat'].isin(['Substance_use_status', 'Sdoh_status']) & ~df['relation'].isnull()]['concept_cat']
    df.loc[~df['concept_cat'].isin(['Substance_use_status', 'Sdoh_status']) & ~df['relation'].isnull(),'SDoH_concept'] = \
        df.loc[~df['concept_cat'].isin(['Substance_use_status', 'Sdoh_status']) & ~df['relation'].isnull()]['concept_value']
    _dicts = df.loc[df['child_concept_cat'].isin(['Substance_use_status', 'Sdoh_status'])][['id', 'child_concept_value', 'child_context']].to_dict('records')
    for _dict in _dicts:
        value_context = ''
        df.loc[df['id']==_dict['id'],'SDoH_value'] = _dict['child_concept_value']
        #print(_dict['child_context'])
        if str(_dict['child_context']).__contains__('#'):
            df.loc[df['id']==_dict['id'],'context'] = _dict['child_context']
        else:
            old_context = next(iter(df.loc[df['id']==_dict['id'], 'context']))
            value_context = old_context.replace('##', '')
            #print(value_context)
            #print(_dict['child_concept_value'])
            child_value = _dict['child_concept_value']
            try:
                match = re.search(r"{}".format(child_value), r"{}".format(value_context))
            except:
                print(child_value, value_context)
                print(r'{}'.format(child_value))
                #exit()#print(match)
            if match is not None:
                value_context = value_context[0:int(match.start())] +"##"+ value_context[int(match.start()):int(match.end())] + "##" + value_context[int(match.end()):-1]
                df.loc[df['id']==_dict['id'],'context'] = value_context
    df = df[['id', 'SDoH_type', 'SDoH_value', 'SDoH_concept', 'context', 'i_0', 'i_f', 'relation', 'child_concept_cat', 'child_concept_value']]

    # get attributes
    _df = df.loc[df['relation'].isnull()][['id', 'SDoH_type', 'SDoH_value', 'SDoH_concept', 'context']]  # separate roots without child (i.e., no relation) from ones that have
    df = df.loc[~df['relation'].isnull()]
    if len(df):
        df.loc[~(df['child_concept_cat'].isnull()),'SDoH_attributes'] = df.loc[~(df['child_concept_cat'].isnull())][['child_concept_cat', 'child_concept_value']].apply(": ".join, axis=1)
        df.loc[df['child_concept_cat'].isin(['Substance_use_status', 'Sdoh_status']),'SDoH_attributes'] = np.nan # Don't create attribute if child_concept_value is now SDoH_value
        # group attributes that have the same parent
        df = df.drop(columns=['child_concept_cat', 'child_concept_value']).groupby(['id', 'SDoH_type', 'SDoH_value', 'SDoH_concept', 'relation', 'context'], dropna=False)['SDoH_attributes'].apply(lambda x: ', '.join(x.dropna())).reset_index()
        df.loc[df['SDoH_attributes'].str.contains(':'),'SDoH_attributes'] = '{' + df.loc[df['SDoH_attributes'].str.contains(':')]['SDoH_attributes'].astype(str) + '}'
        df['SDoH_attributes'].replace('', np.nan, inplace=True)
        df.loc[~(df['SDoH_attributes'].isnull()),'SDoH_attributes'] = df.loc[~(df['SDoH_attributes'].isnull())][['relation', 'SDoH_attributes']].apply(": ".join, axis=1)
        # combine all the attributes for each root
        df = df.drop(columns=['relation']).groupby(['id', 'SDoH_type', 'SDoH_value', 'SDoH_concept', 'context'], dropna=False)['SDoH_attributes'].apply(lambda x: ', '.join(x.dropna())).reset_index()
        df = pd.concat([_df, df]).drop(columns=['id'])
        df['SDoH_attributes'].replace('', np.nan, inplace=True)
    else:
        df = _df.drop(columns=['id'])
    
    # corner cases (drug_type)
    for k,v in zip(['smok','drug','alcoh'],['Tobacco_use', 'Drug_use', 'Alcohol_use']):
        row_I = df['SDoH_type'].str.contains(k,case=False) & ~df['SDoH_type'].isin(['Tobacco_use', 'Drug_use', 'Alcohol_use'])
        df.loc[row_I, 'SDoH_attributes'] = 'Substance_use_status-' + df.loc[row_I]['SDoH_type'] + ': {' + df.loc[row_I]['SDoH_type'] + ": " + df.loc[row_I]['SDoH_value'] + "}"
        df.loc[row_I, 'SDoH_type'] = v
        df.loc[row_I, 'SDoH_value'] = 'yes'
    
    return df

def helper(brat_list):
    df_out_lst = []
    for counts, brat in enumerate(brat_list): ## change this for N lists of brats?
        txt = brat.parent / (brat.stem + '.txt')
        tup_relation = []
        tup_entity = []    
        tup_unit = []
        with open(brat) as b:
            
            text = path_encoded_text / (brat.stem + '.txt')
            with open(text) as t:
                t_lines = t.read()
                #print(t_lines)
                
            lines = b.readlines()
            
            for line in lines:
                concept_cat = None
                if line.strip().startswith('T'):
                    entity_id, concept_info, concept_value = line.strip().split('\t')
                    #ßprint(concept_info)
                    try:
                        concept_cat, start_idx, end_idx = concept_info.split(' ')
                    except:
                        print(concept_info)
                        break
                    context_start = max(0,int(start_idx)-text_range)
                    context_end = min(int(end_idx)+text_range, len(t_lines))
                    context = t_lines[context_start:int(start_idx)] +"##"+ t_lines[int(start_idx):int(end_idx)] + "##" + t_lines[int(end_idx):context_end]
        
                        
                    tup_entity.append((entity_id, concept_cat, int(start_idx), int(end_idx), concept_value, context))
                elif line.strip().startswith('R'):
                    _, relation_info = line.strip().split('\t')
                    tup_relation.append(tuple([x.split(':')[1] for x in relation_info.split(' ')[1:]] + [relation_info.split(' ')[0]]))
                elif line.strip().startswith('A'):
                    entity_id, concept_info = line.strip().split('\t')
                    if len(concept_info.split(' '))==3:
                        concept_cat, parent_id, concept_value = concept_info.split(' ')
                    elif len(concept_info.split(' '))==2:
                        concept_cat, parent_id = concept_info.split(' ')
                        if concept_cat=='negated':
                            concept_cat, concept_value = ('negation', 'negated')
                        else:
                            raise NotImplementedError()
                    else:
                        raise NotImplementedError()
                    tup_relation.append((parent_id, entity_id, None))
                    context = t_lines[context_start:int(start_idx)] +"##"+ t_lines[int(start_idx):int(end_idx)] + "##" + t_lines[int(end_idx):context_end]
                    tup_entity.append((entity_id, concept_cat, None, None, concept_value, context))
                else:
                    pass # raise NotImplementedError()
                
        df_entity = pd.DataFrame(tup_entity, columns =['id', 'concept_cat', 'i_0', 'i_f', 'concept_value', 'context'])

        df_relation = pd.DataFrame(tup_relation, columns =['parent_id', 'child_id', 'relation'])
        df_entity_child = pd.DataFrame(tup_entity, columns =['child_id', 'child_concept_cat', 'child_i_0', 'child_i_f', 'child_concept_value', 'child_context'])
        df_entity_parent = pd.DataFrame(tup_entity, columns =['parent_id', 'parent_concept_cat', 'parent_i_0', 'parent_i_f', 'parent_concept_value', 'parent_context'])
        df_entity = df_entity.merge(df_relation, left_on='id', right_on='parent_id', how='left').drop(columns=['parent_id']).merge(df_entity_child, on='child_id', how='left')
        df_relation = pd.DataFrame(tup_relation, columns =['parent_id', '_child_id', 'relation']).drop(columns=['relation'])
        df_entity = df_entity.merge(df_relation, left_on='id', right_on='_child_id', how='left').drop(columns=['_child_id']).merge(df_entity_parent, on='parent_id', how='left')

        # 2nd order relation
        df_gchild = df_entity.dropna(subset=['parent_id','child_id'])[['id','child_concept_cat','child_concept_value', 'child_context','child_i_0','child_i_f', 'relation']] # gchild must have parent
        df_gchild.columns = ['_child_id','gchild_concept_cat', 'gchild_concept_value', 'gchild_context','gchild_i_0','gchild_i_f', 'child_relation']
        df_gchild = df_entity.dropna(subset=['child_id']).merge(df_gchild, left_on='child_id', right_on='_child_id', how='inner').drop(columns=['child_id', '_child_id']) # gparent must have child

        if df_gchild.shape[0]:
            df_gchild = df_gchild[['parent_id', 'id', 'concept_cat', 'i_0', 'i_f', 'concept_value', 'context', 'gchild_concept_cat', 'gchild_concept_value', 'gchild_context', 'child_relation']]
            df_gchild.columns = ['parent_id', 'id', 'concept_cat', 'i_0', 'i_f', 'concept_value', 'context', 'child_concept_cat', 'child_concept_value', 'child_context', 'relation']            
            df_out = pd.concat([df_entity[['parent_id', 'id', 'concept_cat', 'i_0', 'i_f', 'concept_value', 'context', 'child_concept_cat', 'child_concept_value', 'relation']],
                                df_gchild])
        else:
            df_out = df_entity[['parent_id', 'id', 'concept_cat', 'i_0', 'i_f', 'concept_value', 'context', 'child_concept_cat', 'child_concept_value', 'child_context', 'relation']]

        df_out = df_out.loc[df_out['parent_id'].isnull()]
        df_out.drop(columns=[x for x in df_out.columns if ('parent_' in x)], inplace=True)

        df_out = gen_adrd_output_df(df_out)        
        df_out['note_id'] = brat.stem
        df_out_lst.append(df_out)

    df_out = pd.concat(df_out_lst)#.merge(meta_df, left_on='note_id', right_on='note_ID', how='left').drop(columns=['note_id'])
    return df_out

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="configuration file")
    parser.add_argument("--experiment", type=str, required=True, help="experiement to run")
    parser.add_argument("--processors", type=str, default='12', help="number of cores to use in parallel")
    # sys_args = ["--config", "/home/jameshuang/Projects/NLP_annotation/params/config.yml", "--experiment", "lungrads_ner_validation_baseline"]
    # sys_args = ["--config", "/home/jameshuang/Projects/NLP_annotation/params/config.yml", "--experiment", "lungrads_pipeline", "--gpu_nodes", "0", "1", "2", "3"]
    # sys_args = ["--config", "/home/jameshuang/Projects/NLP_annotation/params/config.yml", "--experiment", "SDoH_pipeline", "--gpu_nodes", "0", "1", "2", "3", "4"]
    # args = parser.parse_args(sys_args)
    args = parser.parse_args()
    
    with open(Path(args.config), 'r') as f:
        experiment_info = yaml.safe_load(f)[args.experiment]

    text_range = 100

    # path_relation_brat = Path('/home/alexgre/projects/from_wu_server/experiements/2020_lungrads/datasets/training')
    # path_encoded_text = Path('/home/alexgre/projects/from_wu_server/experiements/2020_lungrads/datasets/training')
    path_root          = Path(experiment_info['root_dir'])
    path_relation_brat = path_root / 'brat_re' #Path('/home/dparedespardo/project/SDoH_pipeline_demo/demo_data/brat_re') #Path('/data/datasets/shared_data_2/ADRD/clinical_notes_1/brat_re')
    path_encoded_text  = path_root / 'encoded_text' # Path('/home/dparedespardo/project/SDoH_pipeline_demo/demo_data/encoded_text') #Path('/data/datasets/shared_data_2/ADRD/clinical_notes_1/encoded_text')
    #path_report_meta   = Path('/data/datasets/Tianchen/data_from_old_server/2021/ADRD_data_from_Xi/note_details_0826.csv')
    path_output_csv    = path_root / 'output_csv' # Path('/home/dparedespardo/project/SDoH_pipeline_demo/demo_data/output_csv') #Path('/data/datasets/shared_data_2/ADRD/clinical_notes_1/output_csv')
    os.makedirs(path_output_csv, exist_ok=True)

    N = int(args.processors)
    df_final = pd.DataFrame()
    brat_list = np.array_split(list(path_relation_brat.glob("*.ann")), N)
    with ProcessPoolExecutor(max_workers=N) as exe:
        for each in exe.map(helper, brat_list):
            df_final = pd.concat([df_final, each], ignore_index=True)
    
    df_final.to_csv(path_output_csv / 'agg_output.csv', index=False)