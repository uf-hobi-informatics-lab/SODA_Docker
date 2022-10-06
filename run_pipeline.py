from pathlib import Path
from NLPreprocessing.annotation2BIO import generate_BIO, pre_processing, BIOdata_to_file, read_annotation_brat
from NLPreprocessing.text_process.sentence_tokenization import SentenceBoundaryDetection
# from ClinicalTransformerNER.src.run_transformer_batch_prediction import multiprocessing_wrapper, argparser, main
from ClinicalTransformerNER.src.transformer_ner.transfomer_log import TransformerNERLogger
from ClinicalTransformerNER.src.common_utils.output_format_converter import main as format_converter
from ClinicalTransformerNER.src.run_transformer_batch_prediction import main as run_ner
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

def get_subdirs(p):
    sub_ps = [(p / x) for x in os.listdir(p)]
    return sum([get_subdirs(x) for x in sub_ps], []) if len(sub_ps) else [p]

# TODO: notes in subdirectories
class BatchProcessor(object):

    def __init__(self, root_dir=None, raw_data_dir=None, device=None, gpu_nodes=None, output=None, batch_sz=None, 
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
        self.output                     = output 
        self.dependency_tree            = defaultdict(list)
        for dependency in dependency_tree:
            self.dependency_tree[dependency[0]].append(dependency[1])
        
        self.sent_tokenizer             = None

        self.encoded_text               = {}
        self.bio_init                   = {}
        self.brat                       = {}
        self.sent_bounds                = {}
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
        self.brat               = {}
        self.sent_bounds        = {}
        self.entities           = {}
        self.relations          = {}
        self.negations          = {}

    def read_encoded_text(ifn):
        with open(ifn, "r") as f:
            txt = f.read()
        return txt

    def read_bio(ifn):
        with open(ifn, "r") as f:
            txt = f.read().strip()

            if len(txt) < 1:
                warnings.warn(f"{ifn} is an empty file.")
                return []

            sents = txt.split("\n\n")
        return sents    

    def read_brat(ifn):
        return read_annotation_brat(ifn, include_id=True)[1]
    
    def get_batch_files(self, subdir, extension='txt'):
        batch_files = []
        for i, file in enumerate(subdir.glob(f'**/*.{extension}')):
            if (i % self.n_gpu_nodes) == self.gpu_idx:
                batch_files.append(file)
            if i % self.batch_sz == (self.batch_sz-1):
                yield batch_files
                batch_files = []
        if batch_files:
            return batch_files

    def load_output(self, output, batch_files):
        
        extention_dict = {'encoded_text': 'txt', 
                          'bio_init'    : 'bio', 
                          'brat'        : 'ann', 
                          'brat_re'     : 'ann'}
        
        if output == 'raw_text':
            return True
        elif output in extention_dict:
            _exist = True
            for file in batch_files:
                output_cache = getattr(self, output)
                if file.stem in output_cache:
                    continue
                if (self._root_dir / output / '.'.join([file.name, extention_dict[output]])).is_file():
                    _load_funct = getattr(self, f'read_{output}')
                    output_cache[file.stem] = _load_funct(file)
                else:
                    _exist = False
            return _exist        
        else:
            raise NotImplementedError()

    def get_encoded_text(self, batch_files):

        for file in batch_files:
            with open(file, "r", encoding='latin') as f:
                txt = f.read()
            txt = unicodedata.normalize("NFKD", ftfy.fix_text(txt)).strip()
            self.encoded_text[file.stem] = txt
            with open (self._root_dir / 'encoded_text' / file.name,'w',encoding="utf-8") as f:
                f.write(txt)

    def get_bio_init(self, batch_files):
        
        if self.sent_tokenizer is None:
            obj_class = globals()[self.sent_tokenizer_params.get('class')]
            self.sent_tokenizer = obj_class()

        for file in batch_files:
            _, sents = pre_processing(file, deid_pattern=self.sent_tokenizer_params['params']['deid_pattern'], sent_tokenizer=self.sent_tokenizer)
            nsents, _ = generate_BIO(sents, [], file_id=file, no_overlap=False)
            self.bio_init[file.stem] = nsents
            BIOdata_to_file(self._root_dir / 'bio_init' / (file.stem + '.bio'), nsents)

    def get_brat(self):
        
        args = Namespace(**self.ner_model_params)
        
        args.device                 = self.device
        args.batch_files            = list(self.bio_init.keys())
        args.sents                  = copy.deepcopy(self.bio_init)
        args.preprocessed_text_dir  = self._root_dir / 'bio_init'
        args.logger                 = TransformerNERLogger(self._root_dir / 'logs' / f"ner_{self.gpu_idx}.log", 'i').get_logger()

        labeled_bio = run_ner(args, return_labeled_bio=True)
        
        self.brat = format_converter(text_dir=str(self._root_dir / 'encoded_text'),
                                    input_bio_dir=str(self._root_dir / 'encoded_text'),
                                    output_dir=str(self._root_dir / 'brat'),
                                    formatter=1, do_copy_text=False, labeled_bio_tup_lst=labeled_bio, use_bio=False, return_dict=True)
        
    def get_output(self, output, batch_files):
        
        _exist = self.load_output(output, batch_files) 
        if not _exist:
            
            for dependency in self.dependency_tree[output]:
                self.get_output(dependency, batch_files)

            funct = getattr(self, f"get_{output}")
            funct(batch_files)            
        return 
    
    def run(self):
    
        for subdir in self.get_subdirs(self.root_dir):
            gap = relpath(str(subdir), self.root_dir)
            self._root_dir = self.root_dir / gap
            
            for batch_files in self.get_batch_files(subdir):
                
                self.get_output(self.output, batch_files)
                self.clear_cache()

class PipelineManager(object):

    def __init__(self, device=None, root_dir=None, raw_data_dir=None, ner_model={}, relation_model={}, negation_model={}, gpu_nodes=(0, 1), 
                 sent_tokenizer={}, batch_sz=1e4, dependency_tree=[], result=None):
        
        self.device             = device
        self.root_dir           = Path(root_dir) if root_dir else None
        self.raw_data_dir       = Path(raw_data_dir) if raw_data_dir else None
        self.ner_model          = ner_model
        self.relation_model     = relation_model
        self.negation_model     = negation_model
        self.sent_tokenizer     = sent_tokenizer
        self.gpu_nodes          = gpu_nodes
        self.batch_sz           = batch_sz
        
        self.result             = result
        
        # Dependency tree as parent-children dictionary
        self.dependency_tree    = defaultdict(list)
        for dependency in dependency_tree:
            self.dependency_tree[dependency[0]].append(dependency[1])
        
    # root_dir: /data/datasets/shared_data_2/IRB201901754_lungrads
    # raw_data_dir: /data/datasets/shared_data_2/IRB201901754_lungrads/raw_text/
    # ner_model:
    # type: roberta
    # path: /home/jameshuang/Projects/NLP_annotation/fine_tuned_ner/mimiciii_roberta-base_10e_128b_bz=8
    # negation_model:
    # type: bert-base

    def run(self):
        pass


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
    parser.add_argument("--result", type=str, default='output', help="result to generate")
    # sys_args = ["--config", "/home/jameshuang/Projects/NLP_annotation/params/config.yml", "--experiment", "lungrads_ner_validation_baseline"]
    # sys_args = ["--config", "/home/jameshuang/Projects/NLP_annotation/params/config.yml", "--experiment", "lungrads_pipeline", "--gpu_nodes", "0", "1", "2", "3"]
    # sys_args = ["--config", "/home/jameshuang/Projects/NLP_annotation/params/config.yml", "--experiment", "SDoH_pipeline", "--gpu_nodes", "0", "1", "2", "3", "4"]
    # args = parser.parse_args(sys_args)
    args = parser.parse_args()
    
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
    
    