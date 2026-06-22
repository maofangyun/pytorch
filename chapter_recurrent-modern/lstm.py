import torch
from torch import nn
from d2l import torch as d2l
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings

# =====================================================================
# 第一部分：从零开始实现长短期记忆网络 (LSTM)
# =====================================================================

# 1.1 加载时光机器数据集
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 1.2 初始化模型参数
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选记忆元参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 1.3 初始化状态
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

# 1.4 定义 LSTM 模型逻辑
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)

# 1.5 训练从零开始实现的 LSTM
def train_scratch():
    print("开始训练从零开始实现的 LSTM...")
    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    num_epochs, lr = 500, 1
    model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                                init_lstm_state, lstm)
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    plt.show()

# =====================================================================
# 第二部分：简洁实现 (使用 PyTorch nn.LSTM API)
# =====================================================================

def train_concise():
    print("\n开始训练简洁实现的 LSTM...")
    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    num_epochs, lr = 500, 1
    num_inputs = vocab_size
    lstm_layer = nn.LSTM(num_inputs, num_hiddens)
    model = d2l.RNNModel(lstm_layer, len(vocab))
    model = model.to(device)
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    plt.show()

# =====================================================================
# 第三部分：课后练习 5 —— 基于时间序列预测的 LSTM 模型 (北京 PM2.5 预测)
# =====================================================================

# 定义纯原生 PyTorch LSTM 模型用于时间序列预测
class PM25_LSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1):
        super(PM25_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取时间序列的最后一个时间步的输出
        prediction = self.fc(out)
        return prediction

def train_pm25_forecasting():
    warnings.filterwarnings('ignore')  # 忽略一些pandas的弃用警告
    print("\n==========================================")
    print("3. 基于时间序列 (北京 PM2.5) 的预测项目")
    print("==========================================")
    
    print("正在从 UCI 机器学习库下载北京 PM2.5 数据集...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"数据下载失败: {e}，请检查网络连接。")
        return

    # 数据清洗：处理缺失值
    df['pm2.5'] = df['pm2.5'].bfill().ffill()
    df.dropna(inplace=True)

    # 提取关键数值特征
    features = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws']
    dataset = df[features].values

    # 特征缩放
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)

    # 构建滑动窗口 (时间步切分)
    def create_dataset(data, time_step=24):
        X, Y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), :])
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    time_step = 24
    X, Y = create_dataset(dataset_scaled, time_step)

    # 划分训练集和测试集 (前 80% 训练，后 20% 测试)
    train_size = int(len(X) * 0.8)
    X_train_np, Y_train_np = X[:train_size], Y[:train_size]
    X_test_np, Y_test_np = X[train_size:], Y[train_size:]

    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train = torch.tensor(Y_train_np, dtype=torch.float32).unsqueeze(-1)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    Y_test = torch.tensor(Y_test_np, dtype=torch.float32).unsqueeze(-1)

    # 使用 DataLoader 构建批次
    batch_size = 256
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用计算设备: {device}")

    model = PM25_LSTM(input_size=5).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    train_losses = []
    test_losses = []

    print("开始训练 PM2.5 预测模型...")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_X.size(0)
            
        epoch_train_loss /= len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        # 测试评估
        model.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                test_outputs = model(batch_X)
                t_loss = criterion(test_outputs, batch_Y)
                epoch_test_loss += t_loss.item() * batch_X.size(0)
                
        epoch_test_loss /= len(test_dataset)
        test_losses.append(epoch_test_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.5f}, Test Loss: {epoch_test_loss:.5f}')

    # 绘制 Loss 曲线图
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(test_losses, label='Test Loss', color='red', linewidth=2)
    plt.title('Beijing PM2.5 LSTM Forecasting: Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 抽取测试集中最后 200 个小时的真实值与预测值进行曲线对比
    model.eval()
    with torch.no_grad():
        sample_X = X_test[-200:].to(device)
        sample_Y_true = Y_test[-200:].numpy()
        sample_Y_pred = model(sample_X).cpu().numpy()

    plt.figure(figsize=(12, 5))
    plt.plot(sample_Y_true, label='True PM2.5 (Scaled)', color='blue')
    plt.plot(sample_Y_pred, label='Predicted PM2.5 (Scaled)', color='orange', linestyle='dashed')
    plt.title('Test Set Forecasting Result (Last 200 Hours)')
    plt.xlabel('Time Steps (Hours)')
    plt.ylabel('Scaled PM2.5 Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# =====================================================================
# 第四部分：自建字符级语言模型 (Char-Level Language Model)
# =====================================================================

class CharLevelLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1):
        super(CharLevelLanguageModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        logits = self.fc(out)
        return logits

def train_char_language_model():
    print("\n==========================================")
    print("4. 字符级语言模型训练与生成")
    print("==========================================")
    
    raw_text = "hello pytorch, deep learning is fun! " * 20
    chars = sorted(list(set(raw_text)))
    vocab_size = len(chars)

    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    print(f"语料总长度: {len(raw_text)}, 词表大小 (vocab_size): {vocab_size}")

    seq_length = 15
    X_data, Y_data = [], []

    for i in range(0, len(raw_text) - seq_length):
        seq_in = raw_text[i : i + seq_length]
        seq_out = raw_text[i + 1 : i + seq_length + 1]
        X_data.append([char_to_idx[char] for char in seq_in])
        Y_data.append([char_to_idx[char] for char in seq_out])

    X = torch.tensor(X_data, dtype=torch.long)
    Y = torch.tensor(Y_data, dtype=torch.long)

    embed_dim = 16
    hidden_size = 64
    model = CharLevelLanguageModel(vocab_size, embed_dim, hidden_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 100
    print("开始训练语言模型...")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X) 
        logits_flatten = logits.view(-1, vocab_size)
        Y_flatten = Y.view(-1)
        loss = criterion(logits_flatten, Y_flatten)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 文本生成函数
    def generate_text(model, start_string, generate_length=40):
        model.eval()
        input_seq = [char_to_idx[ch] for ch in start_string]
        input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)
        generated_text = start_string
        
        with torch.no_grad():
            for _ in range(generate_length):
                logits = model(input_tensor) 
                last_timestep_logits = logits[:, -1, :] 
                predicted_idx = torch.argmax(last_timestep_logits, dim=1).item()
                predicted_char = idx_to_char[predicted_idx]
                generated_text += predicted_char
                input_seq = input_seq[1:] + [predicted_idx]
                input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)
        return generated_text

    print("\n开始生成文本测试...")
    test_start = "hello"
    print(f"输入提示词: '{test_start}'")
    generated = generate_text(model, start_string=test_start, generate_length=40)
    print(f"模型生成结果: '{generated}'")

# =====================================================================
# 主函数入口
# =====================================================================
if __name__ == "__main__":
    # 可以在这里选择性运行不同的部分：
    
    # 1. 训练从零实现的 LSTM (时光机器数据集)
    train_scratch()
    
    # 2. 简洁实现训练 (时光机器数据集)
    train_concise()
    
    # 3. 时间序列 PM2.5 预测 (练习5)
    train_pm25_forecasting()
    
    # 4. 字符级语言模型
    train_char_language_model()
