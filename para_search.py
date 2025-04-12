from train import *
import itertools
from data_loader import *
import pandas as pd
from tqdm import tqdm

def save_best_model(model, filepath):
    np.save(filepath, model.params)

def hyperparameter_search(param_space, X_train, X_val, y_train, y_val,epochs=100):
    best_val_acc = 0  # 最佳验证集准确率
    best_params = None  # 最佳超参数
    best_model = None  # 最佳模型
    results = []  # 记录每组超参数的结果

    # 获取所有超参数组合
    all_combinations = list(itertools.product(*param_space.values()))

    # 遍历每组超参数组合
    for combination in tqdm(all_combinations, desc="Hyperparameter Search"):
        params = dict(zip(param_space.keys(), combination))  # 把超参数组合成字典

        # 打印当前的超参数组合
        print(f"Training with: {params}")

        # 使用当前超参数训练模型
        model = MLP3(hidden_dim=params['hidden_dim'])  # 动态设置隐藏层大小
        best_model_temp, history = train(
            model, X_train, X_val, y_train, y_val,
            lr=params['learning_rate'],
            batch_size=params['batch_size'],
            epochs=epochs,
            lambda_reg=params['lambda_reg'],
        )

        # 获取当前超参数下最佳验证准确率
        best_val_epoch_acc = max(history['val_accuracy'])  # 取验证集准确率的最大值
        print(f"Best Validation Accuracy for this setup: {best_val_epoch_acc}")

        # 记录当前超参数组合的结果
        results.append((params, best_val_epoch_acc))

        # 如果当前超参数组合的验证集准确率最好，更新最佳模型和超参数
        if best_val_epoch_acc > best_val_acc:
            best_val_acc = best_val_epoch_acc
            best_params = params
            best_model = best_model_temp

    return best_params, best_model, results

if __name__ == '__main__':
    # 设定超参数搜索空间
    param_space = {
        'learning_rate': [0.01,0.1],  # 学习率
        'hidden_dim': [512,256,128],  # 隐藏层大小
        'lambda_reg': [0.0001, 0.001, 0.01],  # L2 正则化强度
        'batch_size': [64, 128, 256],  # 批大小
    }
    X_train, y_train, X_val, y_val,X_test,y_test = get_data()
    model = MLP3()
    best_params, best_model, results = hyperparameter_search(param_space, X_train, X_val, y_train, y_val)

    # 使用超参数搜索后的最佳模型保存
    save_best_model(best_model, "./best_model_params.npy")
    print(best_params)

    # 保存完整搜索结果为 CSV
    df_results = pd.DataFrame(results, columns=['params', 'val_accuracy'])
    df_expanded = df_results['params'].apply(pd.Series)
    df_expanded['val_accuracy'] = df_results['val_accuracy']
    df_expanded.to_csv("./hyperparameter_search_results.csv", index=False)





