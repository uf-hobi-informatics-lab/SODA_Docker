from NLPreprocessing.annotation2BIO import pre_processing, read_annotation_brat, generate_BIO
from NLPreprocessing.text_process.sentence_tokenization import SentenceBoundaryDetection
from collections import defaultdict
from pathlib import Path
from itertools import permutations
import shutil

def create_entity_to_sent_mapping(nnsents, entities, idx2e):
    loc_ens = []
    
    ll = len(nnsents)
    mapping = defaultdict(list)
    for idx, each in enumerate(entities):
        en_label = idx2e[idx]
        en_s = each[2][0]
        en_e = each[2][1]
        new_en = []
        
        i = 0
        while i < ll and nnsents[i][1][0] < en_s:
            i += 1
        s_s = nnsents[i][1][0]
        s_e = nnsents[i][1][1]

        if en_s == s_s:
            mapping[en_label].append(i)

            while i < ll and s_e < en_e:
                i += 1
                s_e = nnsents[i][1][1]
            if s_e == en_e:
                 mapping[en_label].append(i)
            else:
                mapping[en_label].append(i)
                print("last index not match ", each)
        else:
            mapping[en_label].append(i)
            print("first index not match ", each)

            while i < ll and s_e < en_e:
                i += 1
                s_e = nnsents[i][1][1]
            if s_e == en_e:
                 mapping[en_label].append(i)
            else:
                mapping[en_label].append(i)
                print("last index not match ", each)
    return mapping

def get_permutated_relation_pairs(eid2idx):
    all_pairs = []
    all_ids = [k for k, v in eid2idx.items()]
    for e1, e2 in permutations(all_ids, 2):
        all_pairs.append((e1, e2))
    return all_pairs

def gene_neg_relation(perm_pairs, true_pairs, mappings, ens, e2i, nnsents, nsents, valid_comb, fid=None):
    neg_samples = []
    for each in perm_pairs:
        enid1, enid2 = each
        
        # not in true relation
        if (enid1, enid2) in true_pairs:
            continue
        
        enc1 = ens[e2i[enid1]]
        enc2 = ens[e2i[enid2]]

        enbs1, enbe1 = mappings[enid1]
        en1 = nnsents[enbs1: enbe1+1]
        si1, sii1, fs1, ors1 = format_relen(en1, 1, nsents)
        enbs2, enbe2 = mappings[enid2]
        en2 = nnsents[enbs2: enbe2+1]
        si2, sii2, fs2, ors2 = format_relen(en2, 2, nsents)
        sent_diff = abs(si1 - si2)
        
        en1t = en1[0][-1].split("-")[-1]
        en2t = en2[0][-1].split("-")[-1]
        
        if (en1t, en2t) not in valid_comb:
            continue
        
        if sent_diff <= CUTOFF:
            check_tags(fs1, fs2)
            assert (en1t, en2t) in valid_comb, f"{en1t} {en2t}"
            if fid:
                neg_samples.append((sent_diff, NEG_REL, fs1, fs2, en1t, en2t, enid1, enid2, fid))
            else:
                neg_samples.append((sent_diff, NEG_REL, fs1, fs2, en1t, en2t, enid1, enid2))
    
    return neg_samples

def format_relen(en, rloc, nsents):
    if rloc == 1:
        spec1, spec2 = EN1_START, EN1_END
    else:
        spec1, spec2 = EN2_START, EN2_END
    sn1, tn1 = en[0][3]
    sn2, tn2 = en[-1][3]
    target_sent = nsents[sn1]
    target_sent = [each[0] for each in target_sent]
    ors =  " ".join(target_sent)
    
    if sn1 != sn2:
        tt = nsents[sn2]
        tt = [each[0] for each in tt]
        target_sent.insert(tn1, spec1)
        tt.insert(tn2+1, spec2)
        target_sent = target_sent + tt
    else:
        target_sent.insert(tn1, spec1)
        target_sent.insert(tn2+2, spec2)
    
    fs = " ".join(target_sent)
    
    return sn1, sn2, fs, ors

def check_tags(s1, s2):
    assert EN1_START in s1 and EN1_END in s1, f"tag error: {s1}"
    assert EN2_START in s2 and EN2_END in s2, f"tag error: {s2}"

# TODO: wtire everything all at once
def to_tsv(data, fn):
    full_text = ["\t".join([str(i+1) for i in range(len(data[0]))])]
    for each in data:
        full_text.append("\t".join([str(e) for e in each]))
    with open(fn, "w") as f:
        f.write("\n".join(full_text))

def all_in_one(*dd):
    data = []
    for d in dd:
        for k, v in d.items():
            for each in v:
                data.append(each[1:])
    path_tsv.mkdir(parents=True, exist_ok=True)
    
    to_tsv(data, path_tsv / "test.tsv")
            
if __name__ == '__main__':
    
    # Define data/ouput folders
    path_root           = Path('/data/datasets/shared_data_2/ADRD/clinical_notes_1')
    path_encoded_text   = path_root / 'encoded_text'
    path_brat           = path_root / 'brat'
    path_tsv            = path_root / 'tsv'
    path_logs           = path_root / 'logs'
    path_brat_re        = path_root / 'brat_re'
    rel_mappings = []

    MIMICIII_PATTERN = "\[\*\*|\*\*\]"
    EN1_START = "[s1]"  # primary entity starts
    EN1_END = "[e1]"    # primary entity ends
    EN2_START = "[s2]"  # secondary entity starts
    EN2_END = "[e2]"    # secondary entity ends
    NEG_REL = "NonRel"  # Default relation
    
    # TODO: move all parameters to config 
    CUTOFF = 1          # max valid cross sentence distance 
    OUTPUT_CV = False   # output 5-fold cross validation data
    DO_BIN = False      # do binary classification (if false, then we do multiclass classification)    
    sdoh_valid_comb = {
            ('Tobacco_use', 'Substance_use_status'), ('Substance_use_status', 'Smoking_type'),
            ('Substance_use_status', 'Smoking_freq_ppd'), ('Substance_use_status', 'Smoking_freq_py'), 
            ('Substance_use_status', 'Smoking_freq_qy'), ('Substance_use_status', 'Smoking_freq_sy'),
            ('Substance_use_status', 'Smoking_freq_other'), ('Alcohol_use', 'Substance_use_status'),
            ('Substance_use_status', 'Alcohol_freq'), ('Substance_use_status', 'Alcohol_type'), 
            ('Substance_use_status', 'Alcohol_other'), ('Drug_use', 'Substance_use_status'),
            ('Substance_use_status', 'Drug_freq'), ('Substance_use_status', 'Drug_type'),('Substance_use_status', 'Drug_other'), ('Sex_act', 'Sdoh_status'),
            ('Sex_act', 'Partner'), ('Sex_act', 'Protection'), 
            ('Sex_act', 'Sex_act_other'), ('Occupation', 'Employment_status'),
            ('Occupation', 'Employment_location'), ('Gender', 'Sdoh_status'),('Social_cohesion', 'Social_method'), ('Social_method', 'Sdoh_status'),
            ('Physical_act', 'Sdoh_status'), ('Physical_act', 'Sdoh_freq'), 
            ('Living_supply', 'Sdoh_status'), ('Abuse', 'Sdoh_status'),
            ('Transportation', 'Sdoh_status'), ('Health_literacy', 'Sdoh_status'),
            ('Financial_constrain', 'Sdoh_status'), ('Social_cohesion', 'Sdoh_status'),
            ('Social_cohesion', 'Sdoh_freq'), ('Gender', 'Sdoh_status'), 
            ('Race', 'Sdoh_status'), ('Ethnicity', 'Sdoh_status'),
            ('Living_Condition', 'Sdoh_status')
        }               # Allowed relation (i.e., entity category pair)
    preds = defaultdict(list)

    # Create tsv file as dictionary
    sent_tokenizer = SentenceBoundaryDetection()
    for counts, txt_fn in enumerate(path_encoded_text.glob("*.txt")):
        ann_fn = path_brat / (txt_fn.stem + ".ann")

        if not ann_fn.is_file():
            continue
        # TODO: The code below can be further simplified. All we need is sentence boundary, brat, and encoded text to create tsv
        pre_txt, sents = pre_processing(txt_fn, deid_pattern=MIMICIII_PATTERN, sent_tokenizer=sent_tokenizer)
        e2i, ens, _ = read_annotation_brat(ann_fn)
        i2e = {v: k for k, v in e2i.items()}
        
        nsents, sent_bound = generate_BIO(sents, ens, file_id="", no_overlap=False, record_pos=True)
        total_len = len(nsents)
        nnsents = [w for sent in nsents for w in sent]
        mappings = create_entity_to_sent_mapping(nnsents, ens, i2e)
        
        perm_pairs = get_permutated_relation_pairs(e2i)
        pred = gene_neg_relation(perm_pairs, set(), mappings, ens, e2i, nnsents, nsents, sdoh_valid_comb, fid=txt_fn.stem)
        for idx, pred_s in enumerate(pred):
            preds[pred_s[0]].append(pred_s)
        
    # save tsv file to path_tsv
    all_in_one(preds)

    # Run relation extraction
    from ClinicalTransformerRelationExtraction.src.relation_extraction import argparser as relation_argparser
    from ClinicalTransformerRelationExtraction.src.relation_extraction import app as run_relation_extraction

    sys_args = {'--model_type': 'bert',
    '--data_format_mode': '0',
    '--classification_scheme': '2',
    '--pretrained_model': 'bert-large',
    '--data_dir': str(path_tsv),
    '--new_model_dir': '/data/datasets/zehao/sdoh/relations_model/bert',
    '--predict_output_file': str(path_tsv / 'predictions.txt'),
    '--overwrite_model_dir': None,
    '--seed': '13',
    '--max_seq_length': '512',
    '--num_core': '10',
    '--cache_data': None,
    '--do_predict': None,
    '--do_lower_case': None,
    '--train_batch_size': '4',
    '--eval_batch_size': '4',
    '--learning_rate': '1e-5',
    '--num_train_epochs': '3',
    '--gradient_accumulation_steps': '1',
    '--do_warmup': None,
    '--warmup_ratio': '0.1',
    '--weight_decay': '0',
    '--max_num_checkpoints': '0',
    '--log_file': str(path_logs / 'log_re.txt')}
    sys_args = sum([([k, v] if not isinstance(v, list) else [k]+v) if (v is not None) else [k] for k,v in sys_args.items()],[])

    args = relation_argparser(sys_args)
    run_relation_extraction(args)

    # Update brat
    from ClinicalTransformerRelationExtraction.src.data_processing.post_processing import argparser as post_processing_argparser
    from ClinicalTransformerRelationExtraction.src.data_processing.post_processing import app as run_post_processing

    sys_args = {'--mode': 'mul',
    '--predict_result_file': str(path_tsv / 'predictions.txt'),
    '--entity_data_dir': str(path_brat),
    '--test_data_file': str(path_tsv / 'test.tsv'),
    '--brat_result_output_dir': str(path_brat_re),
    '--log_file': str(path_logs / 'log_re.txt')}

    sys_args = sum([([k, v] if not isinstance(v, list) else [k]+v) if (v is not None) else [k] for k,v in sys_args.items()],[])

    args = post_processing_argparser(sys_args)
    run_post_processing(args)
