# numpy-mlp-cifar10

用**Numpy**从零开始构建三层神经网络分类器，在CIFAR10数据集上实现图像分类

## 项目构成
`data_loader` ：实现CIFAR-10数据集的自动化下载、预处理及迭代器封装

`model.py` ：三层全连接神经网络（输入层-隐藏层-输出层），手动实现反向传播算法，支持自定义隐藏层维度，可选激活函数；

`utils.py` ：部分激活函数及其导数实现，one-hot编码，评估指标计算(acc)；

`train.py` ：
+ **训练​部分主要逻辑:**

  优化器：随机梯度下降；

  学习率调度：指数衰减策略；

  损失函数：交叉熵损失+L2惩罚项；

  早停机制（Early Stopping）：基于验证集准确率。

+ **输出：**

    训练过程损失/准确率曲线可视化

    自动保存最优模型参数(.npy格式)


`para_search.py` ：参数查找，网格化搜索，参数空间包括包括学习率，隐藏层个数，批量大小，正则化强度等

`test.py` ：加载预训练模型参数，计算测试集分类准确率

## 模型训练
```
python train.py
```



**训练过程可视化**

![loss_acc_dual_axis](https://github.com/user-attachments/assets/4c119055-331e-4057-816f-00b763a3eaf2)


## 测试：
模型权重下载：

百度网盘链接: https://pan.baidu.com/s/15d7HrnC_sVJ3wYPl0OcA0g 提取码: 12fh 

```
python test.py
```




