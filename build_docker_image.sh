#!/bin/bash

git clone --branch general_purpose_pipeline https://github.com/uf-hobi-informatics-lab/pipeline_dev.git

cd pipeline_dev
git submodule update --init --recursive
cd ..

cp -r ~/models/SDOH_bert_final .
cp -r ~/models/bert .

docker build --no-cache -t sdoh_pipeline:latest .

rm -rf ./pipeline_dev ./SDOH_bert_final ./bert