## Multi-document summarization

#### How to train
```bash
DATA_DIR=/path/to/data
MODEL_DIR=/path/to/pretrained
OUTPUT_DIR=/path/to/output
python singlesum_trainer.py \
--data_dir ${DATA_DIR} \
--model_name_or_path ${MODEL_DIR} \
--tokenizer_name ${VOCAB_FILE} \ 
--output_dir ${OUTPUT_DIR} \
--train_batch_size 4
```

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

#### CPU
```bash
python singlesum_trainer.py
```

#### Multiple-GPUs
```bash
python singlesum_trainer.py --gpus 4
```

