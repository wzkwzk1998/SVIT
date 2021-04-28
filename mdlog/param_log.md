# 4.14

## Xview:

| Time  | Model | Temporal_stride | Dim  | Depth | Heads | Mlp-dim | Dropout | optimizer | base_lr | weight_decay | train_batchsize | test_batchsize | epoch | Top-1  | Top-5  |      |
| ----- | ----- | --------------- | ---- | ----- | ----- | ------- | ------- | --------- | ------- | ------------ | --------------- | -------------- | :---- | ------ | ------ | ---- |
| 18:00 | 1     | 2               | 512  | 6     | 8     | 512     | 0.5     | Adam      | 0.0012  | 0.0001       | 128             | 128            | 80    | 78.57% | 95.94% |      |



# 4.15

## Xview:

| Time  | Model1 | Temporal_stride | Dim  | Depth | Heads | Mlp-dim | Dropout | optimizer | base_lr | weight_decay | train_batchsize | test_batchsize | epoch | Top-1  | Top-5  |
| ----- | ------ | --------------- | ---- | ----- | ----- | ------- | ------- | --------- | ------- | ------------ | --------------- | -------------- | ----- | ------ | ------ |
| 22:00 | 1      | 2               | 512  | 6     | 8     | 512     | 0.5     | Adam      | 0.0012  | 0.0001       | 128             | 128            | 300   | 80.35% | 96.06% |





# 4.22

## Xsub:

| Time  | Model1 | Temporal_stride | Dim  | Depth | Heads | Mlp-dim | Dropout | optimizer | base_lr | weight_decay | lr_scheduler  | T_max | train_batchsize | test_batchsize | epoch | Top-1  | Top-5  |
| ----- | ------ | --------------- | ---- | ----- | ----- | ------- | ------- | --------- | ------- | ------------ | ------------- | ----- | --------------- | -------------- | ----- | ------ | ------ |
| 22:00 | 1      | 2               | 512  | 6     | 8     | 512     | 0.5     | Adam      | 0.0012  | 0.0001       | Cosine linear | 50    | 128             | 128            | 150   | 82.38% | 96.51% |

这一次改变了学习率调整的策略，然后就有了百分之2左右的提升





