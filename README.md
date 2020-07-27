## Document summarization 

#### Requirements
```
pip install -r requirements.txt
```

#### Prepare data 
The data format for training and test is as follows:
```
SOURCE|||||TARGET
```

#### How to train
```bash
DATA_DIR=/path/to/data
MODEL_DIR=/path/to/pretrained
OUTPUT_DIR=/path/to/output
VOCAB_FILE=wp_kt.model3_1
python summarization.py \
--data_dir ${DATA_DIR} \
--model_name_or_path ${MODEL_DIR} \
--tokenizer_name ${VOCAB_FILE} \ 
--output_dir ${OUTPUT_DIR} \
--do_train \
--train_batch_size 4 \
--n_gpu 1 \
```

``DATA_DIR`` should contain two files for training: ``train.txt`` and ``valid.txt``.

``MODEL_DIR`` should contain three files: ``wp_kt.model3_1 (vocab file)``, ``base_model (model file)`` 
and ``bert_config.json (config file)``.
``base_model`` extracted from the original base_model (contains unnecessary parameters) can be downloaded at 
the jbnu server.

#### How to test
```bash
MODEL_DIR=/path/to/pretrained
OUTPUT_DIR=/path/to/output
python test.py \
--input input.txt \ 
--model_name_or_path ${MODEL_DIR} \ 
--output_dir ${OUPUT_DIR} \ 
--device cuda
```

