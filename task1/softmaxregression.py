import numpy as np
import matplotlib.pyplot as plt

class softmax_regression:
    def __init__(self , classes, lr=0.01, epoch=100, batchsize=32):
        self.lr = lr
        self.epoch = epoch
        self.batchsize = batchsize
        self.num_classes = classes
        self.W = None
        self.train_losses = []  # 记录训练集损失
        self.val_losses = []    # 记录验证集损失
        
    def softmax(self, X, Y, X_val, y_val):
        samples, features = X.shape
        self.W = np.zeros((features, self.num_classes))

        for _ in range(self.epoch):
            indices = np.random.permutation(samples)
            for start in range(0, samples, self.batchsize):
                end = min(start + self.batchsize, samples)
                batch_indices = indices[start:end]
                X_batch = X[batch_indices].toarray()
                y_batch = Y.values[batch_indices]
                for i in range(len(batch_indices)):
                    xi = X_batch[i]
                    yi = y_batch[i]
                    scores = np.dot(xi, self.W).reshape(1, -1)  # 将 scores 转换为二维数组
                    exp_scores = np.exp(scores - np.max(scores))  # 数值稳定性
                    probs = exp_scores / np.sum(exp_scores)
                    loss_grad = xi.reshape(-1, 1) @ ((probs - (yi == np.arange(self.num_classes))) / self.batchsize)
                    self.W -= self.lr * loss_grad
                    
            # 在每个 epoch 结束后计算训练集和验证集的损失
            train_loss = self.calculate_loss(X, Y)
            val_loss = self.calculate_loss(X_val, y_val)

            # 将训练集和验证集的损失记录下来
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

    def calculate_loss(self, X, y):
        scores = np.dot(X, self.W)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))  # 数值稳定性
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_probs = probs[range(len(y)), y]
        data_loss = -np.mean(np.log(correct_probs))
        return data_loss

    def predict(self, X):
        scores = np.dot(X, self.W)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))  # 数值稳定性
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)
    
    def plot_losses(self):
        # 可视化训练集和验证集的误差曲线
        plt.plot(range(1, self.epoch + 1), self.train_losses, label='Train Loss')
        plt.plot(range(1, self.epoch + 1), self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()


    