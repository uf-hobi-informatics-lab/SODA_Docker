from pathlib import Path
import pandas as pd
import re, copy
import numpy as np
import os
from pprint import pprint
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# path_relation_brat = Path('/home/alexgre/projects/from_wu_server/experiements/2020_lungrads/datasets/training')
# path_encoded_text = Path('/home/alexgre/projects/from_wu_server/experiements/2020_lungrads/datasets/training')
path_relation_brat = Path('/data/datasets/shared_data_2/ADRD/clinical_notes_1/brat_re')
path_encoded_text = Path('/data/datasets/shared_data_2/ADRD/clinical_notes_1/encoded_text')
path_report_meta = Path('/data/datasets/Tianchen/data_from_old_server/2021/ADRD_data_from_Xi/note_details_0826.csv')
path_output_csv = Path('/data/datasets/shared_data_2/ADRD/clinical_notes_1/output_csv')
os.makedirs(path_output_csv, exist_ok=True)

meta_df = pd.read_csv(path_report_meta)
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
                        ['id', 'concept_cat', 'concept_value', 'child_concept_cat', 'relation'], dropna=False)['child_concept_value'].apply('|'.join).reset_index()])
    
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
    _dicts = df.loc[df['child_concept_cat'].isin(['Substance_use_status', 'Sdoh_status'])][['id', 'child_concept_value']].to_dict('records')
    for _dict in _dicts:
        df.loc[df['id']==_dict['id'],'SDoH_value'] = _dict['child_concept_value']
    df = df[['id', 'SDoH_type', 'SDoH_value', 'SDoH_concept', 'relation', 'child_concept_cat', 'child_concept_value']]

    # get attributes
    _df = df.loc[df['relation'].isnull()][['id', 'SDoH_type', 'SDoH_value', 'SDoH_concept']]  # separate roots without child (i.e., no relation) from ones that have
    df = df.loc[~df['relation'].isnull()]
    df.loc[~(df['child_concept_cat'].isnull()),'SDoH_attributes'] = df.loc[~(df['child_concept_cat'].isnull())][['child_concept_cat', 'child_concept_value']].apply(": ".join, axis=1)
    df.loc[df['child_concept_cat'].isin(['Substance_use_status', 'Sdoh_status']),'SDoH_attributes'] = np.nan # Don't create attribute if child_concept_value is now SDoH_value
    # group attributes that have the same parent
    df = df.drop(columns=['child_concept_cat', 'child_concept_value']).groupby(['id', 'SDoH_type', 'SDoH_value', 'SDoH_concept', 'relation'], dropna=False)['SDoH_attributes'].apply(lambda x: ', '.join(x.dropna())).reset_index()
    df.loc[df['SDoH_attributes'].str.contains(':'),'SDoH_attributes'] = '{' + df.loc[df['SDoH_attributes'].str.contains(':')]['SDoH_attributes'].astype(str) + '}'
    df['SDoH_attributes'].replace('', np.nan, inplace=True)
    df.loc[~(df['SDoH_attributes'].isnull()),'SDoH_attributes'] = df.loc[~(df['SDoH_attributes'].isnull())][['relation', 'SDoH_attributes']].apply(": ".join, axis=1)
    # combine all the attributes for each root
    df = df.drop(columns=['relation']).groupby(['id', 'SDoH_type', 'SDoH_value', 'SDoH_concept'], dropna=False)['SDoH_attributes'].apply(lambda x: ', '.join(x.dropna())).reset_index()
    df = pd.concat([_df, df]).drop(columns=['id'])
    df['SDoH_attributes'].replace('', np.nan, inplace=True)
    
    # corner cases (drug_type)
    for k,v in zip(['smok','drug','alcoh'],['Tobacco_use', 'Drug_use', 'Alcohol_use']):
        row_I = df['SDoH_type'].str.contains(k,case=False) & ~df['SDoH_type'].isin(['Tobacco_use', 'Drug_use', 'Alcohol_use'])
        df.loc[row_I, 'SDoH_attributes'] = 'Substance_use_status-' + df.loc[row_I]['SDoH_type'] + ': {' + df.loc[row_I]['SDoH_type'] + ": " + df.loc[row_I]['SDoH_value'] + "}"
        df.loc[row_I, 'SDoH_type'] = v
        df.loc[row_I, 'SDoH_value'] = 'yes'
    
    return df

text_range = 100

df_out_lst = []
for counts, brat in enumerate(path_relation_brat.glob("*.ann")):
    txt = brat.parent / (brat.stem + '.txt')
    tup_relation = []
    tup_entity = []    
    tup_unit = []
    with open(brat) as b:
        
        text = path_encoded_text / (brat.stem + '.txt')
        with open(text) as t:
            t_lines = t.read()
            
        lines = b.readlines()
        for line in lines:
            concept_cat = None
            if line.strip().startswith('T'):
                entity_id, concept_info, concept_value = line.strip().split('\t')
                concept_cat, start_idx, end_idx = concept_info.split(' ')
                tup_entity.append((entity_id, concept_cat, int(start_idx), int(end_idx), concept_value, 
                                   t_lines[max(0,int(start_idx)-text_range):min(int(end_idx)+text_range,len(t_lines))]))
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
                tup_entity.append((entity_id, concept_cat, None, None, concept_value, None))
            else:
                raise NotImplementedError()
            
    df_entity = pd.DataFrame(tup_entity, columns =['id', 'concept_cat', 'i_0', 'i_f', 'concept_value', 'context'])
    if tup_relation:
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
            df_gchild = df_gchild[['parent_id', 'id', 'concept_cat', 'i_0', 'i_f', 'concept_value', 'context', 'gchild_concept_cat', 'gchild_concept_value', 'child_relation']]
            df_gchild.columns = ['parent_id', 'id', 'concept_cat', 'i_0', 'i_f', 'concept_value', 'context', 'child_concept_cat', 'child_concept_value', 'relation']            
            df_out = pd.concat([df_entity[['parent_id', 'id', 'concept_cat', 'i_0', 'i_f', 'concept_value', 'context', 'child_concept_cat', 'child_concept_value', 'relation']],
                                df_gchild])
        else:
            df_out = df_entity[['parent_id', 'id', 'concept_cat', 'i_0', 'i_f', 'concept_value', 'context', 'child_concept_cat', 'child_concept_value', 'relation']]

        df_out = df_out.loc[df_out['parent_id'].isnull()]
        df_out.drop(columns=[x for x in df_out.columns if ('parent_' in x) or ('context' in x) or ('i_' in x)], inplace=True)
        df_out = gen_adrd_output_df(df_out)        
        df_out['note_id'] = int(brat.stem.split('_')[0])
        df_out_lst.append(df_out)
        
    if counts == 499:
        break

df_out = pd.concat(df_out_lst).merge(meta_df, left_on='note_id', right_on='note_ID', how='left').drop(columns=['note_id'])
df_out.to_csv(path_output_csv / 'test_output.csv', index=False)