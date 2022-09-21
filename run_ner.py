# -*- coding: utf-8 -*-
from pathlib import Path
from NLPreprocessing.annotation2BIO import generate_BIO, pre_processing, BIOdata_to_file
from NLPreprocessing.text_process.sentence_tokenization import SentenceBoundaryDetection
from ClinicalTransformerNER.src.run_transformer_batch_prediction import multiprocessing_wrapper, argparser, main
from ClinicalTransformerNER.src.transformer_ner.transfomer_log import TransformerNERLogger
MIMICIII_PATTERN = "\[\*\*|\*\*\]"
import torch.multiprocessing as mp
import unicodedata, os
import argparse, torch
import cProfile, yaml, copy
from encode_text import preprocessing

# encode raw textdef 
def add_subdir_to_path(p,subdir):
    if subdir is not None:
        return p.parent / subdir / p.name
    else:
        return p

def encode_raw_text(_source_path, _encoded_path):
    for root, _, files in os.walk(_source_path):
        for file in files:
            if not file.endswith(".txt"):
                continue
            txt_fn = Path(root) / file
            if os.stat(txt_fn).st_size == 0:
                continue
            with open(txt_fn,'r',encoding="utf-8") as f:
                txt = unicodedata.normalize("NFKD", f.read()).strip()
            with open (_encoded_path / file, 'w', encoding="utf-8") as f:
                f.write(txt)

# generate bio
def encoded_txt_to_bio(encoded_path, bio_path, subdirs):
    sent_tokenizer = SentenceBoundaryDetection()
    for subdir in subdirs:
        _encoded_path = add_subdir_to_path(encoded_path,subdir)
        _bio_path = add_subdir_to_path(bio_path,subdir)
        for root, _, files in os.walk(_encoded_path):
            for file in files:
                if not file.endswith(".txt"):
                    continue
                txt_fn = Path(root) / file
                bio_fn = _bio_path / (txt_fn.stem + ".bio.txt")
                
                _, sents = pre_processing(txt_fn, deid_pattern=MIMICIII_PATTERN, sent_tokenizer=sent_tokenizer)
                nsents, _ = generate_BIO(sents, [], file_id=txt_fn, no_overlap=False)
                
                BIOdata_to_file(bio_fn, nsents)

# run ner prediction
def run_ner_pred(sys_args, subdirs):
    sys_args = sum([([k, v] if not isinstance(v, list) else [k]+v) if (v is not None) else [k] for k,v in sys_args.items()],[])
    args = argparser(sys_args)
    args.subdirs = subdirs
    if args.gpu_nodes is not None:
        multiprocessing_wrapper(args)
    else:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        logger = TransformerNERLogger(args.log_file, args.log_lvl).get_logger()
        args.logger = logger
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Task will use cuda device: GPU_{}.".format(torch.cuda.current_device())
                    if torch.cuda.device_count() else 'Task will use CPU.')
        main(args)

def get_subdirs(p):
    sub_ps = [(p / x) for x in os.listdir(p)]
    return sum([get_subdirs(x) for x in sub_ps], []) if len(sub_ps) else [p]
    
def get_subdir(root_path, subdir_path):
    subdir_recursive = []
    while subdir_path != root_path:
        subdir_recursive.insert(0, subdir_path.name)
        subdir_path = subdir_path.parent
    return Path('/'.join(subdir_recursive))
    
def run(experiment_info):

    # pr = cProfile.Profile()
    # pr.enable()
    
    test_root = Path(experiment_info['root_dir'])
    source_path = Path(experiment_info['raw_data_dir'])
    generate_bio = experiment_info.get('generate_bio', False)
    encoded_text = experiment_info.get('encoded_text', False)
    ner_model = experiment_info['ner_model']
    
    #  Check if encoded text exist
    test_roots = [x.parent for x in test_root.rglob("**/") if x.name == "encoded_text"]
    if not(encoded_text and len(test_roots)):
        # Generate encoded text if it doesn't exist
        preprocessing(source_path, test_root)
        test_roots = [x.parent for x in test_root.rglob("**/") if x.name == "encoded_text"]

    if args.gpu_nodes is not None:
        mp.set_start_method('spawn')

    if not test_roots:
        subdirs = [None]
    else:
        subdirs = [get_subdir(test_root, copy.deepcopy(x)) for x in test_roots]
        
    encoded_path = test_root / "encoded_text"
    
    # Create output/log path
    suffix = experiment_info.get('suffix', '')
    log_path = test_root / ("_".join(["logs", suffix]) if suffix else "logs") 
    pred_brat_path = test_root / ("_".join(["brat", suffix]) if suffix else "brat")

    # Run NER prediction
    sys_args_dict = {"--model_type":ner_model['type'],\
                    "--pretrained_model":ner_model['path'],\
                    "--raw_text_dir":str(encoded_path),\
                    "--output_dir_brat": str(pred_brat_path),\
                    "--max_seq_length":"128",\
                    "--do_lower_case":None,\
                    "--eval_batch_size":"8",\
                    "--log_file":str( log_path / "ner{}{}.log".format(("_" if suffix else ""), suffix)),\
                    "--do_format":"1",\
                    "--data_has_offset_information":None}

    # Specify gpu nodes if defined
    if experiment_info["gpu_nodes"] is not None:
        sys_args_dict["--gpu_nodes"] = experiment_info["gpu_nodes"]

    # Run NER prediction
    if generate_bio:
        bio_path = test_root / "bio_init"

        encoded_txt_to_bio(encoded_path, bio_path, subdirs)
        
        pred_bio_path = test_root / "bio"

        sys_args = copy.deepcopy(sys_args_dict)
        sys_args.update({"--preprocessed_text_dir":str(bio_path),\
                        "--output_dir":str(pred_bio_path),\
                        "--do_copy": None})
        run_ner_pred(sys_args, subdirs)
    else:

        sys_args = copy.deepcopy(sys_args_dict)
        sys_args.update({"--preprocessed_text_dir":str(encoded_path),\
                        "--output_dir":'',
                        "--no_bio": None})
        run_ner_pred(sys_args, subdirs)

    # pr.disable()
    # pr.dump_stats(log_path / f'{suffix}.profile')

# Example:
# python run_ner.py --config ./config.yml --experiment SDoH_pipeline --gpu_nodes 0 1 2 3 4
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="configuration file")
    parser.add_argument("--experiment", type=str, required=True, help="experiement to run")
    parser.add_argument("--gpu_nodes", nargs="+", default=None, help="gpu_device_id")
    # sys_args = ["--config", "/home/jameshuang/Projects/NLP_annotation/params/config.yml", "--experiment", "lungrads_ner_validation_baseline"]
    # sys_args = ["--config", "/home/jameshuang/Projects/NLP_annotation/params/config.yml", "--experiment", "lungrads_pipeline", "--gpu_nodes", "0", "1", "2", "3"]
    # sys_args = ["--config", "/home/jameshuang/Projects/NLP_annotation/params/config.yml", "--experiment", "SDoH_pipeline", "--gpu_nodes", "0", "1", "2", "3", "4"]
    # args = parser.parse_args(sys_args)
    args = parser.parse_args()
    
    # Load configuration
    with open(Path(args.config), 'r') as f:
        experiment_info = yaml.safe_load(f)[args.experiment]
    experiment_info['gpu_nodes'] = args.gpu_nodes
    # Main function
    run(experiment_info)