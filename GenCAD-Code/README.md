---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: deepcad_id
    dtype: string
  - name: cadquery
    dtype: string
  - name: token_count
    dtype: int64
  - name: prompt
    dtype: string
  - name: hundred_subset
    dtype: bool
  splits:
  - name: train
    num_bytes: 828284247.471
    num_examples: 147289
  - name: test
    num_bytes: 44333434.75
    num_examples: 7355
  - name: validation
    num_bytes: 48396543.916
    num_examples: 8204
  download_size: 725034338
  dataset_size: 921014226.137
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
  - split: validation
    path: data/validation-*
---
