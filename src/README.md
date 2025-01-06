# SegBrats

基于BraTS数据集的分割方案


---

## 实现进度
- [ ] 数据集下载脚本  
- [x] 数据集解压脚本 [./utils/unTarfile.py](./utils/unTarfile.py)
- [x] 数据集划分脚本 [./utils/splitDataList.py](./utils/splitDataList.py)
- [x] 数据集加载脚本 [./readDatasets](./readDatasets)

- [x] 训练脚本 [./train.py](./train.py)
- [x] 滑窗推理脚本 [./inference.py](./inference.py)


## Running

```bash
git clone https://github.com/Helium-327/Helium-327-SegBrats.git

cd Helium-327-SegBrats

pip install -r requirements.txt

```

- 数据集目录软链接映射

```shell
ln -s <数据集目录> ./brats21
```

- 全量训练

```shell
    python train.py --bs 4 --train_mode full --valCropSize 128,128,128 --tb True --epochs 400

```

- 少量训练

```shell
    python BraTS_worflow/train.py --bs 2 --epochs 100 --lr 0.0005 --train_mode local --local_train_length <训练集长度> --local_val_length <验证集长度>
```
