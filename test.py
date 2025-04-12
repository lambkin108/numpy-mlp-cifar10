from train import *
from model import *


def test(model_train,X_test,y_test):
    y_pred,_ = model_train.forward(X_test)
    test_acc = accuracy(y_pred, y_test)
    return test_acc
# 加载模型参数
def load_best_model(filepath):
    model = MLP3()  # 初始化模型
    model.params = np.load(filepath, allow_pickle=True).item()  # 加载并赋值给模型的 params
    return model

# 使用保存的最佳模型参数
if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = get_data()
    best_model = load_best_model("./best_model_params.npy")
    test_acc = test(best_model, X_test, y_test)

    print(test_acc)




