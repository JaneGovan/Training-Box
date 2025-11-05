# Training Box
![Training Box](./Training-box.png)
### 训练大语言模型的工具箱（预训练、微调、强化学习）
Toolkit for Training Large Language Models (`Pretraining`, `Fine-tuning`, `RL`)

## 安装依赖项
```python
cd Training-Box
conda create -n train_box python=3.10 -y && conda activate train_box
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 准备数据集
![**数据示例**](./data)

## 训练
### 训练参数
![**配置文件**](./yaml)
### 训练
```python
CUDA_VISIBLE_DEVICES=0 python sft.py
```
