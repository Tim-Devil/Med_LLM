The hierarchy of this proj is displayed below:

```
.
├── data
│   ├── CT_RATE
│   │   └── dataset
│   │       ├── train_fixed
│   │       │   ├── train_1
│   │       │   │   └── train_1_a
│   │       │   │       ├── train_1_a_1.nii.gz
│   │       │   │       └── train_1_a_2.nii.gz
│   │       │   ├── train_10
│   │       │   │   └── train_10_a
│   │       │   │       ├── train_10_a_1.nii.gz
│   │       │   │       └── train_10_a_2.nii.gz
...
│   │       │   └── train_10016
│   │       │       └── train_10016_a
│   │       │           └── train_10016_a_1.nii.gz
│   │       └── ts_seg
│   │           └── ts_total
│   │               └── train_fixed
│   │                   ├── train_1
│   │                   │   └── train_1_a
│   │                   │       ├── train_1_a_1.nii.gz
│   │                   │       └── train_1_a_2.nii.gz
│   │                   ├── train_10
│   │                   │   └── train_10_a
│   │                   │       ├── train_10_a_1.nii.gz
│   │                   │       └── train_10_a_2.nii.gz
...
│   │                   └── train_10016
│   │                       └── train_10016_a
│   │                           └── train_10016_a_1.nii.gz
│   └── get_1w.py
├── model
├── processing_output_pic
│   ├── resize_comparison_train_100_a_1.png
│   ├── resize_comparison_train_10_a_1.png
│   └── resize_comparison_train_1_a_1.png
├── README
├── src
│   ├── data.py
│   ├── _data_test.py
│   ├── main.py
│   └── visual.py
├── test
└── test.ipynb
```

`data` and `model` folder are not in github repo. You can download dataset manually.

Be notified that modifying data path in `main.py` is necessary!!!

- src/data.py
  Designed for dataset load.
- src/_data.test.py
  If you are bewildered by the hierarchy of dataset, you can use this script to test if data.py is able to load data correctly.
- src/visual.py
  visualize module used by main.py.
- main.py
  main program. `--visual` is an optional parameter to visualize pic while program running.
  You may not need it because the pic will be saved. Do not forget to change saving path in main.py

To run this program, you should:

1. change const parameter in main.py.
2. run program: `python src/main.py` or `python src/main.py --visual`

> In `requirements.txt` I do not include pytorch, since it is CUDA-sensitive and...No need for more word.
> 
> Be very cautious because monai package is very very possible to conflict with other package so conda is recommended and you should install monai first!

