from pathlib import Path
from NLPreprocessing.annotation2BIO import generate_BIO, pre_processing, BIOdata_to_file, read_annotation_brat
from NLPreprocessing.text_process.sentence_tokenization import SentenceBoundaryDetection
# from ClinicalTransformerNER.src.run_transformer_batch_prediction import multiprocessing_wrapper, argparser, main
from ClinicalTransformerNER.src.transformer_ner.transfomer_log import TransformerNERLogger
from ClinicalTransformerNER.src.common_utils.output_format_converter import bio2output as format_converter
from ClinicalTransformerNER.src.run_transformer_batch_prediction import main as run_ner
from ClinicalTransformerRelationExtraction.src.relation_extraction import app as run_relation_extraction
MIMICIII_PATTERN = "\[\*\*|\*\*\]"
import torch.multiprocessing as mp
import unicodedata, os, ftfy
import argparse, torch
import cProfile, yaml, copy, inspect, warnings
from encode_text import preprocessing
from collections import defaultdict
from os.path import relpath
from argparse import Namespace

# , note_mod_idx, root_dir=None, gpu_node=0, result='output'

OUTPUT_DIR = ['raw_text', 'encoded_text', 'bio_init', 'brat', 'tsv', 'brat_re', 'brat_neg', 'meta', 'brat_postproc', 'csv_output']

# TODO: notes in subdirectories
class BatchProcessor(object):

    def __init__(self, root_dir=None, raw_data_dir=None, device=None, gpu_nodes=None, result=None, batch_sz=None, 
                 ner_model={}, relation_model={}, negation_model={}, sent_tokenizer={}, dependency_tree=[]):
  
        self.device                     = device
        self.root_dir                   = Path(root_dir) if root_dir else None
        self.raw_data_dir               = Path(raw_data_dir) if raw_data_dir else None
        self.batch_sz                   = batch_sz
        self.ner_model_params           = ner_model
        self.relation_model_params      = relation_model
        self.negation_model_params      = negation_model
        self.sent_tokenizer_params      = sent_tokenizer
        self.gpu_idx                    = gpu_nodes[0]
        self.n_gpu_nodes                = gpu_nodes[1]
        self.result                     = result 
        self.dependency_tree            = defaultdict(list)
        for dependency in dependency_tree:
            self.dependency_tree[dependency[0]].append(dependency[1])
        
        self.sent_tokenizer             = None

        self.encoded_text               = {}
        self.bio_init                   = {}
        self.sent_bounds                = {}
        self.brat                       = {}
        self.entities                   = {}
        self.relations                  = {}
        self.negations                  = {}

        # Init tokenizer
        # obj_class = globals()[sent_tokenizer.get('class')]
        # self.sent_tokenizer = obj_class()
        
        # Init ner model
        
        # Init relation model

        # Init negation model
    def clear_cache(self):
                
        self.encoded_text       = {}
        self.bio_init           = {}
        self.sent_bounds        = {}
        self.brat               = {}
        self.entities           = {}
        self.relations          = {}
        self.negations          = {}

    # @property
    # def sent_bounds(self):
    #     if self.bio_init:
    #         for k, v in self.bio_init.items():
    #             if k in self._sent_bounds:
    #                 continue
    #             else:
    #                 self._sent_bounds[k] = [(sent[0][1][0], sent[-1][1][1]) for sent in v]
    #         return self._sent_bounds
    #     else:
    #         return {}

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
                          'brat_re'     : 'ann'}
        
        if result == 'raw_text':
            return True
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
            self.sent_bounds[ifn.stem] = [(sent[0][1][0], sent[-1][1][1]) for sent in sents]
        return sents

    def read_brat(self, ifn):
        return read_annotation_brat(ifn, include_id=True)[1]

    def read_tsv(self, ifn):
        with open(ifn, "r", encoding="utf-8") as f:
            lines = f.readlines()[1:]
            return [tuple(line.split("\t")[:-1]) for line in lines]
        
    def get_encoded_text(self, batch_files):
        (self._root_dir / 'encoded_text').mkdir(parents=True, exist_ok=True)
        
        for file in batch_files:
            with open(file, "r", encoding='latin') as f:
                txt = f.read()
            txt = unicodedata.normalize("NFKD", ftfy.fix_text(txt)).strip()
            self.encoded_text[file.stem] = txt
            with open (self._root_dir / 'encoded_text' / file.name,'w',encoding="utf-8") as f:
                f.write(txt)

    def get_bio_init(self, batch_files):
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
            BIOdata_to_file(self._root_dir / 'bio_init' / (file.stem + '.bio'), nsents)

    def get_brat(self, batch_files):
        (self._root_dir / 'brat').mkdir(parents=True, exist_ok=True)
        
        args = Namespace(**self.ner_model_params['params'])
        
        args.device                 = self.device
        args.batch_files            = batch_files
        args.preprocessed_text_dir  = self._root_dir / 'bio_init'
        args.progress_bar           = False     # This field is required
        args.logger                 = TransformerNERLogger(self._root_dir / 'logs' / f"ner_{self.gpu_idx}.log", 'i').get_logger()

        labeled_bio = run_ner(args, return_labeled_bio=True, sents=self.bio_init, raw_text=self.encoded_text) # TODO: don't use deepcopy if not necessary
        
        format_converter(str(self._root_dir / 'encoded_text'),
                        str(self._root_dir / 'encoded_text'),
                        str(self._root_dir / 'brat'),
                        "{}\t{} {} {}\t{}", False, labeled_bio_tup_lst=labeled_bio, write_output=True, use_bio=False, return_dict=self.brat)

    def get_tsv(self, batch_files):
        
        def to_tsv(data, fn):
            full_text = ["\t".join([str(i+1) for i in range(len(data[0]))])]
            for each in data:
                full_text.append("\t".join([str(e) for e in each]))
            with open(fn, "w") as f:
                f.write("\n".join(full_text))

        (self._root_dir / 'tsv').mkdir(parents=True, exist_ok=True)
        # ann_id, t_type, offset_s, offset_e, entity_words
        CUTOFF      = self.relation_model_params['params']['CUTOFF']
        EN1_START   = self.relation_model_params['params']['EN1_START']
        EN1_END     = self.relation_model_params['params']['EN1_END']
        EN2_START   = self.relation_model_params['params']['EN2_START']
        EN2_END     = self.relation_model_params['params']['EN2_END']
        NEG_REL     = self.relation_model_params['params']['NEG_REL']
        
        valid_comb = [tuple(x) for x in self.relation_model_params['params']['valid_comb']]
        
        for file in batch_files:
            
            _brat = copy.deepcopy(self.brat[file.stem])
            _brat = {x[0]: (x + (next((i for i, b in enumerate(self.sent_bounds[file.stem]) if (x[2] >= b[0] and x[3] <= b[0]))),)) for x in _brat}
            valid_pairs = [(k_x, k_y) for k_x, x in _brat.items() for k_y, y in _brat.items() if ((x[1], y[1]) in valid_comb and abs(x[-1]-y[-1])<=CUTOFF)]
            _test_data = []
            for eid_1, eid_2 in valid_pairs:
                
                sidx_1 = _brat[eid_1][-1]
                sidx_2 = _brat[eid_2][-1]
                
                widx_1 = next((i for i, w in enumerate(self.bio_init[file.stem][sidx_1]) if w[1][0]==_brat[eid_1][2]))
                widx_2 = next((i for i, w in enumerate(self.bio_init[file.stem][sidx_2]) if w[1][0]==_brat[eid_2][2]))

                sent_1 = " ".join([self.bio_init[file.stem][sidx_1]].insert(widx_1, EN1_START).insert(widx_1+2, EN1_END))
                sent_2 = " ".join([self.bio_init[file.stem][sidx_2]].insert(widx_2, EN2_START).insert(widx_2+2, EN2_END))
                
                _test_data.append((NEG_REL, sent_1, sent_2, _brat[eid_1][1], _brat[eid_2][1], eid_1, eid_2))

            self.tsv[file.stem] = _test_data

            to_tsv([x + (k,) for k,v in _test_data.itesm() for x in v], self._root_dir / 'tsv' / (file.stem + '.tsv'))


#   --brat_result_output_dir $final_brat_output_with_NER_RE
    def get_result(self, result, batch_files):
        
        _exist = self.load_result(result, batch_files) 
        if not _exist:
            
            for dependency in self.dependency_tree[result]:
                self.get_result(dependency, batch_files)

            funct = getattr(self, f"get_{result}")
            funct(batch_files)            
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

    sys_args = ["--config", "/home/jameshuang/Projects/pipeline_dev/pipeline_config.yml", "--experiment", "lungrads_pipeline", "--result", "brat", "--batch_sz", "501"]
    args = parser.parse_args(sys_args)
    # args = parser.parse_args()
    
    # Load configuration
    with open(Path(args.config), 'r') as f:
        experiment_info = yaml.safe_load(f)[args.experiment]
        experiment_info['result'] = args.result
    experiment_info['batch_sz'] = args.batch_sz

    if len(args.gpu_nodes) > 1:
        multiprocessing_wrapper(args, experiment_info)        
    else:
        # Main function
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_nodes[0])
        experiment_info['gpu_nodes'] = (0, 1)
        experiment_info['device'] = torch.device("cuda")
        pipeline = BatchProcessor(**experiment_info)
        pipeline.run()
    
    