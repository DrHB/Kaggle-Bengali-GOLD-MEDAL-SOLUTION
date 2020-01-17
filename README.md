# Kaggle-Bengali
## Setup

### 17 JAN 2019 9:40 AM

Images: Preprocessing seems to make model stuck around local and lb `0.969`
        If we dont preprocess images atleads to higher local score abd lb sc
### EXP_200.ipynb
```
MODEL:           se_resnext50_32x4d
BS:              1024
SZ:              128 (1 CH)
VALID:           5 FOLD CV (FOLD=2)
TFMS:            transform(get_transforms(do_flip=False,max_warp=0.2, max_zoom=1.1, max_rotate=5, 
                 xtra_tfms=[cutout(n_holes=(10,25), length=(10, 30), p=.5)]), size=(SZ, SZ), 
                 resize_method=ResizeMethod.SQUISH, padding_mode='reflection')
MixUP:           True

PRETRAINED:      IMAGENET
NORMALIZE:       ([0.0692], [0.2051])

LOSS:            WEIGHTED [0.7, 0.1, 0.2]
TRAINING:        OPT: Over9000
                 fit_one_cycle(100, lr, wd=1e-2,  pct_start=0.0,  div_factor=100)
                 
NOTEBOOK:        EXP_200 
MODEL WEIGHTS:   [EXP_200_RESNEX_1CH_MISH_SIMPLE_ORIG_2_2.pth]
MODEL TRN_LOSS:  0.597414
MODEL VAL_LOSS:  0.071885
ACCURACY ALL  :  0.982190
LB SCORE:        0.9745 (SUB_NAME: EXP_80_SERESNET101_1CH(version 23/23))

```

## Model Structure
`Mish` only for tails (body was with `nn.ReLU()`)<br/>
n - on the last linear layers `out_feature`s depending on 3 classes [168, 11, 7]

```
  (head1): Head(
    (fc): Sequential(
      (0): AdaptiveConcatPool2d(
        (ap): AdaptiveAvgPool2d(output_size=1)
        (mp): AdaptiveMaxPool2d(output_size=1)
      )
      (1): Mish()
      (2): Flatten()
      (3): BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (4): Dropout(p=0.2, inplace=False)
      (5): Linear(in_features=4096, out_features=512, bias=True)
      (6): Mish()
      (7): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): Dropout(p=0.2, inplace=False)
      (9): Linear(in_features=512, out_features=n, bias=True)
 ```

Comments: Pretrained model trained just OLD DATA gives pretty good results


Conclusion:    Defenetly Imporvement of the model
```
Thing to try:  - Loss without weights
               - GeM polling layer 
               - Full Mish Model                        
 ```
               
