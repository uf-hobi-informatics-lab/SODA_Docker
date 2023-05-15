# FROM conda-ci-linux-64-python3.8:latest
FROM continuumio/miniconda3

COPY ./pipeline_dev /pipeline_dev

COPY ./SDOH_bert_final /models/SDOH_bert_final  

COPY ./bert /models/bert

WORKDIR /pipeline_dev

RUN conda env update --name base --file environment.yml --prune

RUN python -m spacy download en_core_web_sm

ENTRYPOINT ["python", "run_pipeline.py", "--config", "/pipeline_dev/pipeline_config.yml", "--experiment", "sdoh_pipeline_docker", "--raw_data_dir", "/raw_text", "--root_dir", "/output", "--result", "csv_output"]