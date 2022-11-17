# Social Determinants of Health pipeline execution:
## Setting up this repository
After cloning the repository, please run the following commands:
```
git checkout SDoH_pipeline
git submodule init
git submodule update
```
## Needed files and folder structure:
1. encoded_text folder:
   * This folder will contain the relevant text files to be run through the pipeline in plain text format. The location of this folder must be under the root directory specified in the `config.yml` file. 
2. config.yml file:
    * `gpu_node`: Specify the GPUs to be used during the NER and relation extraction steps (NER supports multi-GPU processing. This paramenter can also be overridden when using the bash script.gi)
    * `root dir`: Base directory where output is to be placed by the pipeline. This should be the directory containing your `encoded_text` folder.
    * `raw_data_dir`: Base location of all relevant raw data, to be used if `encoded_text` folder is not provided.
    * `generate_bio`: Defines wheter or not the NER part of the pipeline generates .bio format output.
    * `encoded_text`: Signals if encoded text already exists.
    * `ner_model`: Contains the specific information pointing to the model to be used.
        * `type`: Specify the type of model to train/use.
        * `path`: The location of the pretrained model to be used as a base 

## Example `config.yml`:
```  
config.yml:  
  gpu_node: 4
  root_dir: /home/dparedespardo/project/SDoH_pipeline_demo
  raw_data_dir: /data/datasets/Tianchen/data_from_old_server/2021/ADRD_data_from_Xi/clinical_notes_all_0826/
  generate_bio: False
  encoded_text: True
  ner_model:
    type: bert
    path: /data/datasets/zehao/sdoh/model/SDOH_bert_final
```
## Running the pipeline demo:
To run the demo, please execute the `run_demo.sh` providing the following arguments:
* `-c` (configuration): Takes the path to and name of the relevant configuration file.
* `-e` (experiment): The specific parameters from the `config.yml` file to run our experiment.
* `-n` (nodes): GPU node to be used during the NER and relation extraction segments.
* `-p` (processors): Number of cores to be used during the post-processing step.

### Example:
``` 
./run_demo.sh -c config.yml -n 0 2 4
```

## Output:
The output file, in .csv format, is organized in the following way:
* `SDoH_type`: Internal definition that specifies a Social Determinant of Health
* `SDoH_value`: The real-text extracted value associated to the `SDoH type`.
* `SDoH_concept`: Specific concept related to a `SDoH type`
* `SDoH_attributes`: Further information tied to each entry.
* `note_ID`: The note ID from which this information was extracted.

If you have a metadata file to map individual notes to patients, you can also include the further entries on your output file, such as:
* `deid_pat_ID`: De-identified patient ID (usually associated to multiple notes.)
* `note_type`: Origin of the note (e.g. letter, progress report)
* `ENCNTR_EFF_DATE`: Date associated to the note.
