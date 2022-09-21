import ftfy, unicodedata, os
from pathlib import Path

blacklist = ["_file_names.txt"]

def get_path(path):
    if not isinstance(path, Path):
        assert isinstance(path, str), "input path is neither Path or str"
        return Path(path)
    else:
        return path

def preprocessing(source_dir, output_dir):
    source_dir = get_path(source_dir)
    output_dir = get_path(output_dir)
    for root, _, files in os.walk(source_dir):
        _root_path = Path(root)
        subdirs_lst = []
        while _root_path != source_dir:
            subdirs_lst.insert(0, _root_path.name)
            _root_path = _root_path.parent
        if subdirs_lst:
            dest_path = output_dir / Path(*tuple(subdirs_lst)) / 'encoded_text'
        else:
            dest_path = output_dir / 'encoded_text'
        dest_path.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            if file.endswith(".txt") and not any([file.endswith(x) for x in blacklist]):
                with open(Path(root) / file, "r", encoding='latin') as f:
                    txt = f.read()
                txt = unicodedata.normalize("NFKD", ftfy.fix_text(txt)).strip()
                
                with open (dest_path / file,'w',encoding="utf-8") as f:
                    f.write(txt)
                    
if __name__ == '__main__':
    
    # source_dir = '/home/alexgre/projects/from_wu_server/experiements/2020_lungrads/datasets/all_order_narratives_impressions'
    # output_dir = '/data/datasets/shared_data_2/IRB201901754_lungrads'
    source_dir = '/data/datasets/Tianchen/data_from_old_server/2021/ADRD_data_from_Xi/clinical_notes_all_0826'
    output_dir = '/data/datasets/shared_data_2/ADRD'
    preprocessing(source_dir, output_dir)