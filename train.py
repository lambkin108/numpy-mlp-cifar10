from data_loader import *
from model import *
import matplotlib.pyplot as plt
from utils import *

def loss(y_pred,y_true,params,lamda):
    """
    损失函数：交叉熵损失+L2正则化
    """
    m = y_true.shape[0]  # 样本数量
    epsilon = 1e-8  # 防止log(0)的极小值
    # 计算交叉熵损失
    cross_entropy = -np.sum(y_true * np.log(y_pred + epsilon)) / m
    l2_penalty= 0.5 * lamda* (
        np.sum(params['W1']**2) +  # 第一层权重平方和
        np.sum(params['W2']**2)    # 第二层权重平方和
    )
    total_loss = cross_entropy+l2_penalty
    return total_loss

def sgd(params, grads, lr):
    """随机梯度下降优化器"""
    for param_name in params:
        # 确保每个参数都有对应的梯度
        assert param_name in grads, f"Missing gradient for parameter {param_name}"
        params[param_name] -= lr * grads[param_name]

    return params

def train(model, X_train, X_val, y_train, y_val,
          lr=0.01, batch_size=128, epochs=100,
          lambda_reg=0.001, lr_decay=0.95, patience=5):
    """
    完整训练流程
    返回:
        best_model: 最佳模型参数
        history: 训练历史记录 包含训练集验证集损失数组，验证集准确率数组，用于绘图
    """
    # 初始化训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    # 打印训练前准确率
    y_p_val, _ = model.forward(X_val)
    val_l = loss(y_p_val, y_val, model.params, lambda_reg)
    auc_val = accuracy(y_p_val, y_val)
    print(f"训练前验证集损失 = {val_l}")
    print(f"训练前准确率 = {auc_val}")

    print("---------- 开始训练！！！ ------------")

    best_model = model.params
    best_val_loss = float('inf')
    epochs_without_improvement = 0  # 用于提前停止

    for epoch in range(1, epochs + 1):
        train_iter = data_iter(batch_size, X_train, y_train)
        for X, y in train_iter:
            y_pred, cache = model.forward(X)
            l = loss(y_pred, y, model.params, lambda_reg)
            grads = model.backward(y_pred, y, cache, lambda_reg)
            model.params = sgd(model.params, grads, lr)
        # 计算训练集损失
        y_p, _ = model.forward(X_train)
        train_l = loss(y_p, y_train, model.params, lambda_reg)
        history['train_loss'].append(train_l)
        print(f"第{epoch}轮训练train_l = {train_l}")

        # 验证
        y_p_val, _ = model.forward(X_val)
        val_l = loss(y_p_val, y_val, model.params, lambda_reg)
        auc_val = accuracy(y_p_val, y_val)

        history['val_loss'].append(val_l)
        history['val_accuracy'].append(auc_val)

        print(f"第{epoch}轮验证集损失 val_l = {val_l}")
        print(f"第{epoch}轮准确率 = {auc_val}")

        # 如果当前验证集损失最小，保存模型
        if val_l < best_val_loss:
            best_val_loss = val_l
            best_model = model.params
            epochs_without_improvement = 0  # 重置提前停止计数
        else:
            epochs_without_improvement += 1

        # 学习率衰减
        lr *= lr_decay
        if lr < 1e-5:
            lr = 1e-5

        # 提前停止
        if epochs_without_improvement >= patience:
            print(f"提前停止：验证集损失在连续 {patience} 轮没有改善")
            break

    # 返回最佳模型参数和训练历史记录
    model.params = best_model


    return model, history

def visualization(history):
    """
    双y轴 结果可视化
    训练，验证部分loss曲线
    验证部分准确率曲线

    """
    train_losses = history['train_loss']
    val_losses = history['val_loss']
    val_accs = history['val_accuracy']
    # 示例数据
    epochs = list(range(1, len(train_losses)+1))
    # 创建图像
    fig, ax1 = plt.subplots(figsize=(8, 5))
    # 左 y 轴：绘制 loss 曲线
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue',
             linewidth=2)
    ax1.plot(epochs, val_losses, label='Validation Loss', color='orange',
             linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # 右 y 轴：绘制 accuracy 曲线
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_accs, label='Validation Accuracy', color='green',
             linewidth=2, linestyle='--')
    ax2.set_ylabel('Accuracy')
    ax2.tick_params(axis='y')
    ax2.legend(loc='lower right')

    # 整体布局
    plt.title('Training & Validation Loss + Validation Accuracy')
    fig.tight_layout()
    plt.savefig("./res_pictures/loss_acc_dual_axis.png")

if __name__ == '__main__':
    X_train, y_train, X_val, y_val,X_test,y_test = get_data()
    model = MLP3(hidden_dim=512)
    lr = 0.05
    batch_size = 256
    epochs = 100
    lambda_reg = 0.001

    best_model,history = train(model, X_train, X_val, y_train, y_val,
          lr=lr, batch_size=batch_size, epochs=epochs,
          lambda_reg=lambda_reg)
    def save_best_model(model, filepath):
        np.save(filepath, model.params)
    # 保存验证集上的最优模型
    save_best_model(best_model, "./best_model_params.npy")
    # 可视化训练过程
    visualization(history)




