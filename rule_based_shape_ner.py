from pathlib import Path
from NLPreprocessing.annotation2BIO import read_annotation_brat
from collections import Counter

# Check shape text category
extension = 'ann'
brat = dict()
shape_text_list = []
for subdir in [Path('/home/jameshuang/Projects/lungrads/ner_training/datasets/training/brat'), Path('/home/jameshuang/Projects/lungrads/ner_training/datasets/test/brat')]:
    for file in subdir.glob(f'**/*.{extension}'):
        ents = read_annotation_brat(file, include_id=True)[1]
        shape_text_list.extend([x[1].lower() for x in ents if x[-1] == 'shape'])
        
Counter(shape_text_list)
# 'curvilinear':2
# 'linear':15
# 'peripheral':1
# 'bilateral':6
# 'peripheral irregular':1
# 'spiculated':4
# 'round':3
# 'triangular':1
# 'ovoid':1
# 'lingular':1
# 'tree-in-bud':1
keywords = ['curvilinear', 'linear', 'peripheral', 'bilateral', 'peripheral irregular', 'spiculated', 'round', 'triangular', 'ovoid', 'lingular', 'tree-in-bud']


# for batch_file in batch_files:
#     k = batch_file.stem
#     update_brat = False
#     for i, x in enumerate(self.brat.get(k, [])):
#         eid, ann_text, offset_s, offset_e, label = x
#         if label == 'size':
#             t_lines = self.encoded_text[k]
#             words = t_lines[int(offset_e):].strip().split(' ')
#             unit = next((re.compile('[^a-zA-Z]').sub('', x).lower() for x in words[:5] if re.compile('[^a-zA-Z]').sub('', x).lower() in ['cm', 'mm']), None) # TODO: A x B unit, < A unit
            
#             if unit != normalized_unit:
#                 try:
#                     assert unit in ['cm'], "unit not in ['mm', 'cm'] is not implemented"
#                     unit = 'mm'
#                     self.brat[k][i] =(eid, scale_num(ann_text, 10), offset_s, offset_e, label)
#                     update_brat = True
#                 except:
#                     unit = 'unknown'
#                     pass
            
#             self.brat_unit[k].append(("unit", eid, unit))

