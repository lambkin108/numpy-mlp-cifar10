# numpy-mlp-cifar10

用**Numpy**从零开始构建三层神经网络分类器，实现图像分类

## 项目构成
`model.py` ：模型代码，支持反向传播，自定义隐藏层个数，可选择激活函数；

`utils.py` ：部分激活函数实现，one-hot编码等；

`train.py` ：训练部分主要逻辑，包括SGD 优化器、学习率下降、交叉熵损失和 L2 正则化，根据验证集指标自动保存最优的模型权重，训练过程可视化等；

`para_search.py` ：参数查找，网格化搜索，包括学习率，隐藏层个数，批量大小，正则化强度等

`test.py` ：导入训练好的模型进行测试，输出在测试集上的分类准确率


## 模型训练

`python train.py`



**训练过程可视化**

![loss_acc_dual_axis](https://github.com/user-attachments/assets/4c119055-331e-4057-816f-00b763a3eaf2)


## 测试：
模型权重下载：

百度网盘链接: https://pan.baidu.com/s/15d7HrnC_sVJ3wYPl0OcA0g 提取码: 12fh 


`python test.py`




