## Match summarization

#### How to train
```bash
DATA_DIR=/path/to/data
MODEL_DIR=/path/to/pretrained
OUTPUT_DIR=/path/to/output
--data_dir ../../data/match --model_name_or_path ../../../resources/bert-kt-large --gpus 2
```

#### CPU
```bash
python matchsum_trainer.py
```

#### Multiple-GPUs
```bash
python matchsum_trainer.py --gpus 2
```
