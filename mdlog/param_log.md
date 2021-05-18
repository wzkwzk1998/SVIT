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



# 4.29
## Xview:
   这次的模型，在训练完成之后又更改学习率为0.0002, dropout取0.4进行了微调，比微调之前上涨了3%。
   之后取学习率为0.00005，dropout取0.35，模型的训练集loss上涨的很恐怖，但是测试集loss下降，比较奇怪。

| Time  | Model | Temporal_stride | Dim  | Depth | Heads | Mlp-dim | Dropout | optimizer | base_lr | weight_decay | lr_scheduler | T_max | train_batchsize | test_batchsize | epoch | Top-1  | Top-5  |      |
| ----- | ----- | --------------- | ---- | ----- | ----- | ------- | ------- | --------- | ------- | ------------ | --------------- | -------------- | :---- | ------ | ------ | ---- | :--- | ---- |
| 20:00 | multi-scale     | 2 and 10               | 512  | 6     | 8     | 512     | 0.5     | Adam      | 0.0012  | 0.0001       | Cosine linear | 50     | 128             | 128            | 150   | 81.01% | 95.83% |      |



# 4.30

## Xview:

​	这次是改动dropout， 看看不设置dropout对模型的表现有什么影响

| Time  | Model       | Temporal_stride | Dim  | Depth | Heads | Mlp-dim | Dropout | optimizer | base_lr | weight_decay | train_batchsize | test_batchsize | epoch | Top-1  | Top-5  |      |
| ----- | ----------- | --------------- | ---- | ----- | ----- | ------- | ------- | --------- | ------- | ------------ | --------------- | -------------- | ----- | ------ | ------ | ---- |
| 11:00 | multi-scale | 2 and 10        | 512  | 6     | 8     | 512     | 0.0     | Adam      | 0.0015  | 0.0001       | 128             | 128            | 150   | 84.58% | 96.68% |      |





# 5.10

## Xsub：

用同样的参数在Xsub上训练，效果比较差，推测可能是dropout的原因，过拟合程度较高，但是在xview中不设置dropout模型表现不错，很奇怪。如果将dropout升高，可能加上dropout，xview的表现能够更好。

| Time  | Model       | Temporal_stride | Dim  | Depth | Heads | Mlp-dim | Dropout | optimizer | base_lr | weight_decay | train_batchsize | test_batchsize | epoch | Top-1  | Top-5  |      |
| ----- | ----------- | --------------- | ---- | ----- | ----- | ------- | ------- | --------- | ------- | ------------ | --------------- | -------------- | ----- | ------ | ------ | ---- |
| 22:00 | multi-scale | 2 and 10        | 512  | 6     | 8     | 512     | 0.0     | Adam      | 0.0015  | 0.0001       | 128             | 128            | 150   | 71.68% | 91.03% |      |





# 5.11



## Xsub：

​	这次实验设置了0.5的dropout，然后效果有点差，emmmmmmmm。

| Time  | Model       | Temporal_stride | Dim  | Depth | Heads | Mlp-dim | Dropout | optimizer | base_lr | weight_decay | train_batchsize | test_batchsize | epoch | Top-1  | Top-5  |      |
| ----- | ----------- | --------------- | ---- | ----- | ----- | ------- | ------- | --------- | ------- | ------------ | --------------- | -------------- | ----- | ------ | ------ | ---- |
| 12:00 | multi-scale | 2 and 10        | 512  | 6     | 8     | 512     | 0.5     | Adam      | 0.0015  | 0.0001       | 128             | 128            | 150   | 67.68% | 89.03% |      |





# 5.13

## Xview:

​	Cross, 在调整了学习率之后，模型取得了不错的表现,注意，dropout设置为0.	

| Time  | Model             | Temporal_stride | Dim  | Depth | Heads | Mlp-dim | Dropout | optimizer | base_lr | weight_decay | train_batchsize | test_batchsize | epoch | Top-1  | Top-5  |      |
| ----- | ----------------- | --------------- | ---- | ----- | ----- | ------- | ------- | --------- | ------- | ------------ | --------------- | -------------- | ----- | ------ | ------ | ---- |
| 12:00 | multi-cross-scale | 2 and 10        | 512  | 6     | 8     | 512     | 0.      | Adam      | 0.0005  | 0.0001       | 128             | 128            | 150   | 87.76% | 97.28% |      |



