from utils import *
from torchvision import datasets,transforms
from sklearn.model_selection import train_test_split
import random


def get_data():
    """
    下载CIFAR10(60000×3×32×32)
    将其展平为3072维向量
    训练，验证，测试集比例：4：1：1
    :return:  numpy形式的数据集
    """
    trans_train = transforms.Compose([transforms.ToTensor()])
    trans_test = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(root='../data',train=True,download=True,transform=trans_train)
    test_set = datasets.CIFAR10(root='../data', train=False, download=True,
                                 transform=trans_test)
    X = []
    y = []
    for img,label in train_set:
        X.append(img.numpy())
        y.append(label)
    X = np.array(X)   #(50000, 3, 32, 32)
    y = np.array(y)   #(50000,)
    X = X.reshape(len(X),-1)            #(50000,3072)
    X = (X-0.5)/0.5                 #归一化
    y = to_one_hot(y,num_classes=10) #(50000, 10)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, y_test = [],[]
    for img,label in test_set:
        X_test.append(img.numpy())
        y_test.append(label)
    X_test= np.array(X_test)   #(50000, 3, 32, 32)
    y_test = np.array(y_test)   #(50000,)
    X_test= X_test.reshape(len(X_test),-1)            #(50000,3072)
    X_test= (X_test-0.5)/0.5                 #归一化
    y_test = to_one_hot(y_test,num_classes=10) #(50000, 10)

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, y_train,X_val,y_val,X_test,y_test

def data_iter(batch_size, X, y):
    """
    :param batch_size: 批量大小
    :param X: 特征
    :param y: 目标
    :return: 批量迭代器
    """
    num_examples = len(X)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices =indices[i: min(i + batch_size, num_examples)]
        yield X[batch_indices],y[batch_indices]


if __name__ == '__main__':
    X_train, y_train, X_val, y_val,X_test,y_test = get_data()
    # dataload = data_iter(64,X_train, y_train)
    # x,y= next(dataload)
    # print(x.shape)
    # print(y.shape)
