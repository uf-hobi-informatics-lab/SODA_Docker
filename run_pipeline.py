from pathlib import Path
from NLPreprocessing.annotation2BIO import generate_BIO, pre_processing, BIOdata_to_file, read_annotation_brat
from NLPreprocessing.text_process.sentence_tokenization import SentenceBoundaryDetection
# from ClinicalTransformerNER.src.run_transformer_batch_prediction import multiprocessing_wrapper, argparser, main
from ClinicalTransformerNER.src.transformer_ner.transfomer_log import TransformerNERLogger
from ClinicalTransformerNER.src.common_utils.output_format_converter import bio2output as format_converter
from ClinicalTransformerNER.src.common_utils.output_format_converter import BRAT_TEMPLATE
from ClinicalTransformerNER.src.run_transformer_batch_prediction import main as run_ner

from ClinicalTransformerRelationExtraction.src.utils import TransformerLogger
from ClinicalTransformerRelationExtraction.src.batch_prediction import app as run_relation_extraction
from ClinicalTransformerRelationExtraction.src.data_processing.data_format_conf import BRAT_REL_TEMPLATE
from ClinicalTransformerRelationExtraction.src.data_processing.io_utils import save_text

from ClinicalTransformerClassification.src.batch_prediction import app as negation_classification

MIMICIII_PATTERN = "\[\*\*|\*\*\]"
import pandas as pd
import numpy as np
import torch.multiprocessing as mp
import unicodedata, os, ftfy
import argparse, torch
import re, yaml, copy, warnings, pickle
from encode_text import preprocessing
from collections import defaultdict
from os.path import relpath
from argparse import Namespace

# , note_mod_idx, root_dir=None, gpu_node=0, result='output'

OUTPUT_DIR = ['raw_text', 'encoded_text', 'bio_init', 'brat', 'bio', 'tsv', 'brat_re', 'brat_neg', 'brat_unit', 'csv_output_lungrads']

# TODO: notes in subdirectories
class BatchProcessor(object):

    def __init__(self, root_dir=None, raw_data_dir=None, device=None, gpu_nodes=None, result=None, batch_sz=None, 
                 ner_model={}, relation_model={}, negation_model={}, unit_extraction_model={}, csv_output_params = {},
                 sent_tokenizer={}, dependency_tree=[], debug=True):
  
        self.device                     = device
        self.root_dir                   = Path(root_dir) if root_dir else None
        self.raw_data_dir               = Path(raw_data_dir) if raw_data_dir else None
        self.batch_sz                   = batch_sz
        self.ner_model_params           = ner_model
        self.relation_model_params      = relation_model
        self.negation_model_params      = negation_model
        self.unit_extraction_params     = unit_extraction_model
        self.csv_output_params          = csv_output_params
        self.sent_tokenizer_params      = sent_tokenizer
        self.gpu_idx                    = gpu_nodes[0]
        self.n_gpu_nodes                = gpu_nodes[1]
        self.result                     = result 
        self.dependency_tree            = defaultdict(list)
        for dependency in dependency_tree:
            self.dependency_tree[dependency[0]].append(dependency[1])
        
        self.sent_tokenizer             = None
        self.csv_output                 = []
        self.debug                      = debug
        self.clear_cache()

    def clear_cache(self):
                
        self.encoded_text       = {}
        self.bio_init           = {}
        self.sent_bounds        = {}
        self.brat               = {}
        self.bio                = defaultdict(list)
        self.tsv                = {}
        self.brat_re            = defaultdict(list)
        self.brat_neg           = defaultdict(list)
        self.brat_unit          = defaultdict(list)

    def get_subdirs(self, p):
        sub_ps = [x for x in p.iterdir() if x.is_dir() and x.stem not in OUTPUT_DIR]
        return sum([self.get_subdirs(x) for x in sub_ps], []) if len(sub_ps) else [p]
    
    def get_batch_files(self, subdir, extension='txt'):
        batch_files = []
        for i, file in enumerate(subdir.glob(f'**/*.{extension}')):
            if (i % self.n_gpu_nodes) == self.gpu_idx:
                batch_files.append(file)
            if i % self.batch_sz == (self.batch_sz-1):
                yield batch_files
                batch_files = []
        if batch_files:
            yield batch_files

    def load_result(self, result, batch_files):

        extention_dict = {'encoded_text': 'txt', 
                          'bio_init'    : 'bio', 
                          'brat'        : 'ann',
                          'bio'         : 'bio', 
                          'tsv'         : 'tsv', 
                          'brat_re'     : 'ann', 
                          'brat_neg'    : 'ann', 
                          'brat_unit'   : 'ann'}

        if result == 'raw_text':
            return True
        elif 'csv_output' in result:
            if not self.csv_output:
                _load_funct = getattr(self, f'read_{result}')
                self.csv_output = _load_funct()
            _exist = True
            for file in batch_files:
                if file.stem not in self.csv_output:
                    _exist = False                    
            return _exist
        elif result in extention_dict:
            _exist = True
            for file in batch_files:
                output_cache = getattr(self, result)
                if file.stem in output_cache:
                    continue
                _file = self._root_dir / result / '.'.join([file.stem, extention_dict[result]])
                if _file.is_file():
                    _load_funct = getattr(self, f'read_{result}')
                    output_cache[file.stem] = _load_funct(_file)
                else:
                    _exist = False
            return _exist
        else:
            raise NotImplementedError()

    def read_csv_output_lungrads(self):
            
        output_file = (self._root_dir / 'csv_output' / self.csv_output_params.get('output_file', 'output.csv'))
        processed_file = (self._root_dir / 'csv_output' / 'processed_file.txt')
        if not self.csv_output_params.get('force_rewrite', True) and processed_file.is_file():
            with open(processed_file, "r") as f:
                return f.read().split("\n")
        elif processed_file.is_file():
            processed_file.unlink()
            output_file.unlink()
            return []
        else:
            return []
        
    def read_encoded_text(self, ifn):
        with open(ifn, "r") as f:
            txt = f.read()
        return txt

    def read_bio_init(self, ifn):
        
        def word_formatting(x):
            if len(x) == 5:  # TODO: not sure if it capture all the corner cases
                x.insert(0, '\xa0')
            elif len(x) != 6:
                raise NotImplementedError()
            return [x[0], (int(x[1]), int(x[2])), (int(x[3]), int(x[4])), x[5]]
        
        with open(ifn, "r") as f:
            txt = f.read().strip()

            if len(txt) < 1:
                warnings.warn(f"{ifn} is an empty file.")
                return []

            sents = [[word_formatting(word.split(' ')) for word in sent.split("\n")] for sent in txt.split("\n\n")]
            self.sent_bounds[ifn.stem] = {i: (sent[0][1][0], sent[-1][1][1]) for i, sent in enumerate(sents)}
        return sents

    def read_bio(self, ifn):
        
        return self.read_bio_init(ifn)

    def read_brat(self, ifn):
        return read_annotation_brat(ifn, include_id=True)[1]

    def read_tsv(self, ifn):
        with open(ifn, "r", encoding="utf-8") as f:
            try:
                lines = f.readlines()[1:]
                return [tuple(line.split("\t")[:-1]) for line in lines]
            except:
                return []
        
    def read_brat_re(self, ifn):
        return read_annotation_brat(ifn, include_id=True)[2]
        
    def read_brat_neg(self, ifn):
        return read_annotation_brat(ifn, include_id=True)[3]
    
    def read_brat_unit(self, ifn):
        return read_annotation_brat(ifn, include_id=True)[3]
        
    def get_encoded_text(self, batch_files, write_output):
        if write_output:
            (self._root_dir / 'encoded_text').mkdir(parents=True, exist_ok=True)
        
        for file in batch_files:
            with open(file, "r", encoding='latin') as f:
                txt = f.read()
            txt = unicodedata.normalize("NFKD", ftfy.fix_text(txt)).strip()
            self.encoded_text[file.stem] = txt
            if write_output:
                with open (self._root_dir / 'encoded_text' / file.name,'w',encoding="utf-8") as f:
                    f.write(txt)

    def get_bio_init(self, batch_files, write_output):
        if write_output:
            (self._root_dir / 'bio_init').mkdir(parents=True, exist_ok=True)
        
        if self.sent_tokenizer is None:
            obj_class = globals()[self.sent_tokenizer_params.get('_class')]
            self.sent_tokenizer = obj_class()

        for file in batch_files:
            if file.stem in self.bio_init:
                continue
            _, sents = pre_processing(file, deid_pattern=self.sent_tokenizer_params['params']['deid_pattern'], sent_tokenizer=self.sent_tokenizer)
            nsents, sent_bounds = generate_BIO(sents, [], file_id=file, no_overlap=False)
            self.bio_init[file.stem] = nsents
            self.sent_bounds[file.stem] = sent_bounds    
            if write_output:
                BIOdata_to_file(self._root_dir / 'bio_init' / (file.stem + '.bio'), nsents)

    def get_bio(self, batch_files, write_output):
        if write_output:
            (self._root_dir / 'bio').mkdir(parents=True, exist_ok=True)
        
        for file in batch_files:
            prev_eid = None
            for sent in self.bio_init[file.stem]:
                _sent = []
                for word in sent:
                    eid, tag = next(((ent[0], ent[4]) for ent in self.brat[file.stem] if (word[1][0] >= ent[2]) and (word[1][1] <= ent[3])), (None, None))
                    if eid:
                        prefix = 'B' if eid != prev_eid else 'I'
                        prev_eid = eid
                        _sent.append([word[0], (int(word[1][0]), int(word[1][1])), (int(word[2][0]), int(word[2][1])), f'{prefix}-{tag}'])
                    else:
                        _sent.append(copy.deepcopy(word))
                self.bio[file.stem].append(_sent)

            if write_output:
                BIOdata_to_file(self._root_dir / 'bio' / (file.stem + '.bio'), self.bio[file.stem])                

# [x[0], (int(x[1]), int(x[2])), (int(x[3]), int(x[4])), x[5]]
# (ann_id, t_type, offset_s, offset_e, entity_words)

    def get_brat(self, batch_files, write_output):

        def text2int(textnum):
            # search and replace number words in a string into corresponding digits
            
            numwords={}
            units = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
            ]

            tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

            scales = ["hundred", "thousand", "million", "billion", "trillion"]

            numwords["and"] = (1, 0)
            for idx, word in enumerate(units):  numwords[word] = (1, idx)
            for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
            for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)

            ordinal_words = {'first':1, 'second':2, 'third':3, 'fifth':5, 'eighth':8, 'ninth':9, 'twelfth':12}
            ordinal_endings = [('ieth', 'y'), ('th', '')]

            current = result = 0
            curstring = ""
            onnumber = False
            for word in textnum.split():
                if word in ordinal_words:
                    scale, increment = (1, ordinal_words[word])
                    current = current * scale + increment
                    if scale > 100:
                        result += current
                        current = 0
                    onnumber = True
                else:
                    for ending, replacement in ordinal_endings:
                        if word.endswith(ending):
                            word = "%s%s" % (word[:-len(ending)], replacement)

                    if word not in numwords:
                        if onnumber:
                            curstring = " ".join([curstring, repr(result + current)])
                        curstring = " ".join([curstring, word])
                        result = current = 0
                        onnumber = False
                    else:
                        scale, increment = numwords[word]

                        current = current * scale + increment
                        if scale > 100:
                            result += current
                            current = 0
                        onnumber = True

            if onnumber:
                curstring = " ".join([curstring, repr(result + current)])

            return curstring.lstrip()        
        
        # def drop_label(labeled_bio, label):
        #     for sent in labeled_bio['sents']:
        #         for text, w_s, w_e, w_a_s, w_a_e, predict_tag in sent:
        #             if label in predict_tag:
        #                 sent = (text, w_s, w_e, w_a_s, w_a_e, "O")

        # def keyword_match(labeled_bio, keywords=[], label=None): # Working on this. Need more sophisticated rule to capture relation and reduce FPR
        #     drop_label(labeled_bio, label)
        #     keywords = sorted(keywords, key = lambda x: len(x.split(' ')), reverse=True)
        #     for sent in labeled_bio['sents']:
        #         sent_text = ' '.join([x[0] for x in sent])
        #         word_idx = [0] + list(np.cumsum([len(x)+1 for x in sent_text[:-1]]))
        #         m_starts = []
        #         for keyword in keywords:
        #             for m in re.finditer(r"\b{}\b".format(sent_text), keyword):
        #                 if m.start() not in m_starts:
        #                     m_starts.append(m.start())
        #                     word_idx_start = word_idx.index(m.start())
        #                     prefix = 'B'
        #                     for _word_idx in range(word_idx_start, word_idx_start+len(keyword.split(' '))):
        #                         text, w_s, w_e, w_a_s, w_a_e, _ = sent[_word_idx] 
        #                         sent[_word_idx] = (text, w_s, w_e, w_a_s, w_a_e, f'{prefix}-{label}')
        #                         prefix = 'I'
        
        def postprocess_size(ann_text):

            for old_str, new_str in {'to':'-', re.compile(r"\s?-\s?"):' - '}.items():
                ann_text = re.sub(old_str, new_str, ann_text)
            return text2int(ann_text)
    
        if write_output:
            (self._root_dir / 'brat').mkdir(parents=True, exist_ok=True)
        
        args = Namespace(**self.ner_model_params['params'])
        
        args.device                 = self.device
        args.batch_files            = batch_files
        args.preprocessed_text_dir  = self._root_dir / 'bio_init'
        args.progress_bar           = False     # This field is required
        args.logger                 = TransformerNERLogger(self._root_dir / 'logs' / f"ner_{self.gpu_idx}.log", 'i').get_logger()

        labeled_bio = run_ner(args, return_labeled_bio=True, sents=self.bio_init, raw_text=self.encoded_text) # TODO: don't use deepcopy if not necessary

        # if self.ner_model_params.get("relabel",{}).get("params", {}).get("label", None):
        #     replace_by = self.ner_model_params.get("relabel",{}).get("funct", None)            
        #     _funct = locals()[replace_by]
        #     _funct(labeled_bio, **self.ner_model_params.get("relabel",{}).get("params", {}))
        
        format_converter(str(self._root_dir / 'encoded_text'),
                        str(self._root_dir / 'encoded_text'),
                        str(self._root_dir / 'brat'),
                        "{}\t{} {} {}\t{}", False, labeled_bio_tup_lst=labeled_bio, write_output=False, use_bio=False, return_dict=self.brat)
            
        for batch_file in args.batch_files:
            k = batch_file.stem
            fn = self._root_dir / 'brat' / (k + '.ann')
            if batch_file.stem in self.brat:
                if write_output:
                    save_text("\n".join([BRAT_TEMPLATE.format(eid, label, offset_s, offset_e, (postprocess_size(ann_text) if (label == 'size') else ann_text)) \
                        for eid, ann_text, offset_s, offset_e, label in self.brat[batch_file.stem]]), fn)
            else:
                self.brat[batch_file.stem] = []
                if write_output:
                    with open(self._root_dir / 'brat' / (batch_file.stem + '.ann'), 'w') as f:
                        pass

    def get_tsv(self, batch_files, write_output):
        
        def to_tsv(data, fn):
            if not data:
                with open(fn, "w") as f:
                    pass
                return
            full_text = ["\t".join([str(i+1) for i in range(len(data[0]))])]
            for each in data:
                full_text.append("\t".join([str(e) for e in each]))
            with open(fn, "w") as f:
                f.write("\n".join(full_text))

        if write_output:
            (self._root_dir / 'tsv').mkdir(parents=True, exist_ok=True)
        # ann_id, t_type, offset_s, offset_e, entity_words
        CUTOFF      = self.relation_model_params['preprocess']['CUTOFF']
        EN1_START   = self.relation_model_params['preprocess']['EN1_START']
        EN1_END     = self.relation_model_params['preprocess']['EN1_END']
        EN2_START   = self.relation_model_params['preprocess']['EN2_START']
        EN2_END     = self.relation_model_params['preprocess']['EN2_END']
        NEG_REL     = self.relation_model_params['preprocess']['NEG_REL']
        
        valid_comb = [tuple(x) for x in self.relation_model_params['preprocess']['valid_comb']]
        
        for file in batch_files:
            
            _brat = copy.deepcopy(self.brat[file.stem])
            _brat = [(x + (next((i for i, b in self.sent_bounds[file.stem].items() if (x[2] >= b[0] and x[2] < b[1])),None),)) for x in _brat]
            valid_pairs = [(x, y) for x in _brat for y in _brat if (x[0] != y[0]) and ((x[4], y[4]) in valid_comb and abs(x[-1]-y[-1])<=CUTOFF)]
            _test_data = []
            for _brat_1, _brat_2 in valid_pairs:
                
                sidx_1 = _brat_1[-1]
                sidx_2 = _brat_2[-1]
                
                widx_1 = next((i for i, w in enumerate(self.bio_init[file.stem][sidx_1]) if w[1][0]==_brat_1[2]))
                widx_2 = next((i for i, w in enumerate(self.bio_init[file.stem][sidx_2]) if w[1][0]==_brat_2[2]))

                sent_1 = [x[0] for x in self.bio_init[file.stem][sidx_1]]
                sent_2 = [x[0] for x in self.bio_init[file.stem][sidx_2]]
                
                sent_1.insert(widx_1, EN1_START)
                sent_1.insert(widx_1+2, EN1_END)
                sent_2.insert(widx_2, EN2_START)
                sent_2.insert(widx_2+2, EN2_END)
                
                sent_1 = " ".join(sent_1)
                sent_2 = " ".join(sent_2)
                
                _test_data.append((NEG_REL, sent_1, sent_2, _brat_1[4], _brat_2[4], _brat_1[0], _brat_2[0]))

            self.tsv[file.stem] = _test_data
            if write_output:
                to_tsv([x + (file.stem,) for x in _test_data], self._root_dir / 'tsv' / (file.stem + '.tsv'))

    def get_brat_re(self, batch_files, write_output):

        if write_output:
            (self._root_dir / 'brat_re').mkdir(parents=True, exist_ok=True)
        
        _params = copy.deepcopy(self.relation_model_params['params'])
        _params.update({'device': self.device, 'data_dir': str(self._root_dir / 'tsv'),
                        'logger': TransformerLogger(logger_file=str(self._root_dir / 'logs' / f"re_{self.gpu_idx}.log"), logger_level=_params['log_lvl']).get_logger()
                        })
        
        args = Namespace(**_params)
        tsv = sum(list(self.tsv.values()), [])
        preds = run_relation_extraction(args, tsv=tsv)
        
        type_maps = pickle.load(open(self.relation_model_params['params']['type_map'],'rb'))
        rel_tuple_lst = [(k, type_maps[(ent_1, ent_2)], eid_1, eid_2) for k, v in self.tsv.items() for _, _, _, ent_1, ent_2, eid_1, eid_2 in v]
        
        for (fid, rel_type, eid_1, eid_2), pred in zip(rel_tuple_lst, preds): 
            if pred == self.relation_model_params['params']['non_relation_label']:
                continue
            self.brat_re[fid].append((rel_type, eid_1, eid_2))
        
        for batch_file in batch_files:
            k = batch_file.stem
            fn = self._root_dir / 'brat_re' / (k + '.ann')
            if batch_file.stem in self.brat_re:
                self.brat_re[k] = [(f"R{i+1}",)+x for i, x in enumerate(self.brat_re[k])]    
                if write_output:
                    save_text("\n".join(BRAT_REL_TEMPLATE.format(*((i+1,) + x[1:])) for i, x in enumerate(self.brat_re[k])), fn)
            else:
                self.brat_re[k] = []
                if write_output:
                    with open(fn, 'w') as f:
                        pass

    def get_brat_neg(self, batch_files, write_output):

        if write_output:
            (self._root_dir / 'brat_neg').mkdir(parents=True, exist_ok=True)

        _params = copy.deepcopy(self.negation_model_params['params'])
        _params.update({'device': self.device, 'data_dir': str(self._root_dir / 'tsv'),
                        'logger': TransformerLogger(logger_file=str(self._root_dir / 'logs' / f"neg_{self.gpu_idx}.log"), logger_level=_params['log_lvl']).get_logger()
                        })

        NEG_REL = self.negation_model_params['preprocess']['NEG_REL']

        args = Namespace(**_params)
        tsv_neg = {k: list(set([(NEG_REL, sent_1.replace(self.relation_model_params['preprocess']['EN1_START'], self.negation_model_params['preprocess']['EN1_START']).replace(\
                                                         self.relation_model_params['preprocess']['EN1_END'], self.negation_model_params['preprocess']['EN1_END']), eid_1) \
                                            for _, sent_1, _, _, _, eid_1, _ in v])) for k,v in self.tsv.items()}
        tsv = sum(list(tsv_neg.values()), [])
        preds = negation_classification(args, tsv=tsv)
        
        rel_tuple_lst = [(k, eid_1) for k, v in tsv_neg.items() for _, _, eid_1 in v]
        for (fid, eid_1), pred in zip(rel_tuple_lst, preds): 
            if pred not in self.negation_model_params['postprocess']['keep_cat']:
                continue
            self.brat_neg[fid].append((pred, eid_1))

        for batch_file in batch_files:
            k = batch_file.stem
            fn = self._root_dir / 'brat_neg' / (k + '.ann')
            if k in self.brat_neg:
                self.brat_neg[k] = [(f"A{i+1}",)+x for i, x in enumerate(self.brat_neg[k])]
                if write_output:
                    save_text("\n".join("A{}\t{} {}".format(*((i+1,) + x[1:])) for i, x in enumerate(self.brat_neg[k])), fn)
            else:
                self.brat_neg[k] = []
                if write_output:
                    with open(fn, 'w') as f:
                        pass
        
    def get_brat_unit(self, batch_files, write_output):
    
        if write_output:
            (self._root_dir / 'brat_unit').mkdir(parents=True, exist_ok=True)
        
        def scale_num(input_str, scale): # scale all the digits by "scale"
            test_str_copy = copy.copy(input_str)
            return_str = ""
            for _word in re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", input_str): # Extract all the digits from string
                word = str(float(_word)*scale)
                _idx_fo = test_str_copy.find(_word) + len(_word)
                return_str += test_str_copy[:_idx_fo].replace(_word,word)
                test_str_copy = test_str_copy[_idx_fo:]
            return_str += test_str_copy
            return return_str
        
        normalized_unit = self.unit_extraction_params.get("postprocess",{}).get("normalize",None)

        for batch_file in batch_files:
            k = batch_file.stem
            update_brat = False
            for i, x in enumerate(self.brat.get(k, [])):
                eid, ann_text, offset_s, offset_e, label = x
                if label == 'size':
                    t_lines = self.encoded_text[k]
                    words = t_lines[int(offset_e):].strip().split(' ')
                    unit = next((re.compile('[^a-zA-Z]').sub('', x).lower() for x in words[:5] if re.compile('[^a-zA-Z]').sub('', x).lower() in ['cm', 'mm']), None) # TODO: A x B unit, < A unit
                    
                    if unit != normalized_unit:
                        try:
                            assert unit in ['cm'], "unit not in ['mm', 'cm'] is not implemented"
                            unit = 'mm'
                            self.brat[k][i] =(eid, scale_num(ann_text, 10), offset_s, offset_e, label)
                            update_brat = True
                        except:
                            unit = 'unknown'
                            pass
                    
                    self.brat_unit[k].append(("unit", eid, unit))

            if update_brat:
                fn = self._root_dir / 'brat' / (k + '.ann')
                if write_output:
                    save_text("\n".join([BRAT_TEMPLATE.format(eid, label, offset_s, offset_e, ann_text) for eid, ann_text, offset_s, offset_e, label in self.brat[k]]), fn)

        for batch_file in batch_files:
            k = batch_file.stem
            fn = self._root_dir / 'brat_unit' / (k + '.ann')
            if k in self.brat_unit:
                self.brat_unit[k] = [(f"A{i+1}",)+x for i, x in enumerate(self.brat_unit[k])]
                if write_output:
                    save_text("\n".join("A{}\t{} {} {}".format(*((i+1,) + x[1:])) for i, x in enumerate(self.brat_unit[k])), fn)
            else:
                self.brat_unit[k] = []
                if write_output:
                    with open(fn, 'w') as f:
                        pass

    def get_csv_output_lungrads(self, batch_files, write_output):
        
        if write_output:
            (self._root_dir / 'csv_output').mkdir(parents=True, exist_ok=True)
        
        def get_meta(batch_file):
            irb_id, note_id, note_ver, _, pat_id, date = re.findall(r'(IRB\d+)_(ORDER_\d+)_(\w+)_(IRB\d+)_(PAT_\d+)_(\d+-\d+-\d+)',batch_file.stem)[0]
            return (irb_id, note_id, note_ver, pat_id, date)
        
        df_out_lst = []
        text_range = 100
        for batch_file in batch_files:
            
            tup_relation = []
            tup_entity = []
            _text = self.encoded_text[batch_file.stem]
            
            if not self.brat[batch_file.stem]:
                continue

            tup_entity.extend([(eid, label, ann_text, _text[max(0,int(offset_s)-text_range):min(int(offset_e)+text_range,len(_text))]) \
                for eid, ann_text, offset_s, offset_e, label in self.brat[batch_file.stem]])
            tup_relation.extend([(parent_id, child_id) for _, _, parent_id, child_id in self.brat_re[batch_file.stem]])

            for i, x in enumerate(self.brat_neg[batch_file.stem] + self.brat_unit[batch_file.stem]):
                if len(x) == 3:
                    tup_entity.append((f'A{i+1}', 'negation', x[1], None))
                elif len(x) == 4:
                    tup_entity.append((f'A{i+1}', x[1], x[3], None))
                else:
                    raise NotImplementedError()                
                tup_relation.append((x[2], f'A{i+1}'))
            # tup_entity.extend([(eid, 'negation', pred, None) for (eid, pred, _) in self.brat_neg[batch_file.stem]])
            # tup_entity.extend([(eid, label, ann_text, None) for (eid, label, _, ann_text) in self.brat_unit[batch_file.stem]])                
            # tup_relation.extend([(parent_id, eid) for (eid, _, parent_id) in self.brat_neg[batch_file.stem]])
            # tup_relation.extend([(parent_id, eid) for (eid, _, parent_id, _) in self.brat_unit[batch_file.stem]])

            irb_id, note_id, note_ver, pat_id, date = get_meta(batch_file)
            df_entity = pd.DataFrame(tup_entity, columns =['id', 'concept_cat', 'concept_value', 'context'])
            if tup_relation:
                df_relation = pd.DataFrame(tup_relation, columns =['parent_id', 'child_id'])
                df_entity_child = pd.DataFrame(tup_entity, columns =['child_id', 'child_concept_cat', 'child_concept_value', 'child_context'])
                df_entity_parent = pd.DataFrame(tup_entity, columns =['parent_id', 'parent_concept_cat', 'parent_concept_value', 'parent_context'])
                df_entity = df_entity.merge(df_relation, left_on='id', right_on='parent_id', how='left').drop(columns=['parent_id']).merge(df_entity_child, on='child_id', how='left')
                df_relation = pd.DataFrame(tup_relation, columns =['parent_id', '_child_id'])
                df_entity = df_entity.merge(df_relation, left_on='id', right_on='_child_id', how='left').drop(columns=['_child_id']).merge(df_entity_parent, on='parent_id', how='left')

                # 2nd order relation
                df_gchild = df_entity.dropna(subset=['parent_id'])[['id','concept_cat','concept_value', 'context', 'parent_id']] # gchild must have parent
                df_gchild.columns = ['gchild_id','gchild_concept_cat', 'gchild_concept_value', 'gchild_context', '_parent_id']
                df_gchild = df_entity.dropna(subset=['child_id']).merge(df_gchild, left_on='child_id', right_on='_parent_id', how='inner').drop(columns=['child_id', '_parent_id']) # gparent must have child
                if df_gchild.shape[0]:
                    df_gchild = df_gchild[['id', 'concept_cat', 'concept_value', 'context', 'gchild_concept_cat', 'gchild_concept_value']]
                    df_gchild.columns = ['id', 'concept_cat', 'concept_value', 'context', 'child_concept_cat', 'child_concept_value']            
                    df_out = pd.concat([df_entity.loc[df_entity['concept_cat']=='nodule'][['id', 'concept_cat', 'concept_value', 'context', 'child_concept_cat', 'child_concept_value']],
                                        df_gchild.loc[df_gchild['concept_cat']=='nodule']])
                else:
                    df_out = df_entity.loc[df_entity['concept_cat']=='nodule'][['id', 'concept_cat', 'concept_value', 'context', 'child_concept_cat', 'child_concept_value']]                               
                if df_out.empty:
                    continue
            else:
                df_out = df_entity.loc[df_entity['concept_cat']=='nodule'][['id', 'concept_cat', 'concept_value', 'context']]
                if df_out.empty:
                    continue
                df_out['child_concept_cat'] = np.nan
                df_out['child_concept_value'] = np.nan

            for info, val in zip(['irb_id', 'note_id', 'note_ver', 'pat_id', 'date'], [irb_id, note_id, note_ver, pat_id, date]):
                df_out[info] = val
            df_out_lst.append(df_out)
                
        df_all = pd.concat(df_out_lst, ignore_index=True)
        df_all.loc[~df_all['child_concept_value'].isna(),'child_concept_value'] = df_all.loc[~df_all['child_concept_value'].isna()].groupby(['id','concept_cat','concept_value','context','child_concept_cat','irb_id','note_id','note_ver','pat_id','date'])['child_concept_value'].transform(lambda x: ','.join(x))
        df_all = df_all.drop_duplicates()
        df_all = pd.pivot(df_all, index=['irb_id','note_id','note_ver','pat_id','date','id','concept_cat','concept_value','context'], 
                        columns = 'child_concept_cat',values = 'child_concept_value').reset_index().rename_axis(None, axis=1)

        if write_output:
            
            output_file = self._root_dir / 'csv_output' / self.csv_output_params.get('output_file', 'output.csv')
            if output_file.is_file():
                df_all.to_csv(output_file, mode='a', index=False, header=False)
            else:
                df_all.to_csv(output_file, index=False)

            processed_file = (self._root_dir / 'csv_output' / 'processed_file.txt')
            if processed_file.is_file():
                with open(processed_file, 'a') as f:
                    f.write("\n"+"\n".join([x.stem for x in batch_files]))
            else:
                with open(processed_file, 'w') as f:
                    f.write("\n".join([x.stem for x in batch_files]))

        self.csv_output.extend([x.stem for x in batch_files])

    def get_result(self, result, batch_files):
        
        _exist = self.load_result(result, batch_files) 
        if not _exist:
            
            for dependency in self.dependency_tree[result]:
                self.get_result(dependency, batch_files)

            funct = getattr(self, f"get_{result}")
            _result = (getattr(self, result) if ('csv_output' not in result) else self.csv_output)
            write_output = self.debug or (self.result == result)
            funct([x for x in batch_files if x.stem not in _result], write_output)
        return 
    
    def run(self):
    
        for subdir in self.get_subdirs(self.raw_data_dir):
            gap = relpath(str(subdir), self.raw_data_dir)
            self._root_dir = self.root_dir / gap
            
            for batch_files in self.get_batch_files(subdir):
                
                self.get_result(self.result, batch_files)
                self.clear_cache()


def main(experiment_info):
    
    pipeline = BatchProcessor(**experiment_info)
    pipeline.run()
    
    
def multiprocessing_wrapper(args, experiment_info):
    mp.set_start_method('spawn')
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_nodes))
    
    print("Use GPU devices: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    
    experiment_lst = []
    N_gpu_nodes = len(args.gpu_nodes)
    for i in range(N_gpu_nodes):
        _experiment_info = copy.deepcopy(experiment_info)
        _experiment_info['gpu_nodes'] = (i, N_gpu_nodes)
        _experiment_info['device'] = torch.device("cuda",i)
        experiment_lst.append(_experiment_info)

    with mp.Pool(N_gpu_nodes) as p:
        p.map(main, experiment_lst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sz", type=int, default=1e4, help="batch size")
    parser.add_argument("--config", type=str, required=True, help="configuration file")
    parser.add_argument("--experiment", type=str, required=True, help="experiement to run")
    parser.add_argument("--gpu_nodes", nargs="+", default=[0], help="gpu_device_id")
    parser.add_argument("--result", type=str, default='output_csv', choices=OUTPUT_DIR, help="result to generate")
    parser.add_argument("--debug", type=str, default=True, help="Set True to store intermediate outputs")

    # sys_args = ["--config", "/home/jameshuang/Projects/pipeline_dev/pipeline_config.yml", "--experiment", "lungrads_pipeline", "--result", "csv_output_lungrads", "--batch_sz", "50000", "--gpu_nodes", "2"]
    sys_args = ["--config", "/home/jameshuang/Projects/pipeline_dev/pipeline_config.yml", "--experiment", "lungrads_ner_training", "--result", "bio", "--batch_sz", "1000", "--gpu_nodes", "2"]
    args = parser.parse_args(sys_args)
    # args = parser.parse_args()
    
    # Load configuration
    with open(Path(args.config), 'r') as f:
        experiment_info = yaml.safe_load(f)[args.experiment]
        experiment_info['result'] = args.result
    experiment_info['batch_sz'] = args.batch_sz
    experiment_info['debug'] = args.debug

    if len(args.gpu_nodes) > 1:
        multiprocessing_wrapper(args, experiment_info)        
    else:
        # Main function
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_nodes[0])
        experiment_info['gpu_nodes'] = (0, 1)
        experiment_info['device'] = torch.device("cuda")
        main(experiment_info)
    
    