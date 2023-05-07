###
 # <p>Title:  </p>
 # <p>Create Date: 15:09:05 01/10/22</p>
 # <p>Copyright: College of Medicine </p>
 # <p>Organization: University of Florida</p>
 # @author Yonghui Wu
 # @version 1.0
 # <p>Description: </p>
 ##

import re, copy
from .RuleEngine import RuleEngine
from .TemporalExpression import *

def teExtract(te):
    max_len = 0
    final_te = []
    for expr in te:
        curr_len = len(expr.text)
        if curr_len > max_len:
            max_len = curr_len
            final_te = expr
    return final_te


# if __name__ == "__main__":
def normalization(output_df):

    forbidden_chars = ['?', '*', '[', ']', '@', '(', ')']
    rgx = re.compile('%s' % forbidden_chars)
    # Check for metadata file
    metadata = False

    ''' 0. read dataframe from SDoH pipeline'''
    
    data = copy.deepcopy(output_df)
    data = data.rename(columns={'SDoH_type': 'SDoH_standard_category',
                                'SDoH_concept': 'SDoH_mention',
                                'SDoH_value': 'SDoH_raw_text'})
    data['SDoH_normalized'] = None
    if metadata:
        data = data[['note_id', 'person_id', 'SDoH_standard_category',
                     'SDoH_mention','SDoH_raw_text','SDoH_normalized',
                     'SDoH_attributes','encounter_date']]
    else:
        data = data[['note_id', 'SDoH_standard_category', 'SDoH_mention',
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
        try:
            text = rgx.sub('', str(text).lower())
            results=my_engine.extract(text, data['SDoH_standard_category'][idx])
            
            if results != []:
                results = teExtract(results)
                norm_values.append(results.value)
            else:
                norm_values.append('other')
        except:
            print(f"Skipping over {data['SDoH_standard_category'][idx]}. No rules found.")
            norm_values.append('')
        
    data['SDoH_normalized']=norm_values
    #norm_data = pd.concat([norm_data, current_data])
    print(data)
    data['SDoH_normalized'] = data['SDoH_normalized'].fillna(value='__other__')
    
    return data