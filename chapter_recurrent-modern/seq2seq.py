import collections
import math
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import os

# 解决 Windows 系统下 d2l 库读取英法数据集时的 GBK 编码报错问题 (UnicodeDecodeError)
def read_data_nmt_fixed():
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()
d2l.read_data_nmt = read_data_nmt_fixed

# 解决在非交互式环境下使用 d2l.Animator 报错的问题 (NotImplementedError)
class AnimatorFixed:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        pass

    def add(self, x, y):
        # 仅在命令行打印训练进度
        # 因为 train_seq2seq 每 10 个 epoch 才 add 一次，所以在此直接打印
        y_val = y[0] if isinstance(y, (tuple, list)) else y
        print(f"Epoch {x:3d}: loss = {y_val:.4f}")

d2l.Animator = AnimatorFixed

# =====================================================================
# 第一部分：模型架构定义
# =====================================================================

#@save
class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 门控循环单元 GRU
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 输入 'X' 的形状：(batch_size, num_steps)
        # 输出 self.embedding(X) 的形状：(batch_size, num_steps, embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        # 转换后形状：(num_steps, batch_size, embed_size)
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps, batch_size, num_hiddens)
        # state的形状:(num_layers, batch_size, num_hiddens)
        return output, state


#@save
class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 解码器的输入是将当前步的词元嵌入与上一隐状态（即上下文变量）拼接在一起
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # 编码器输出 enc_outputs 的格式为：(output, state)
        # 我们使用编码器最后一层、最后一个时间步的隐状态来初始化解码器的隐状态
        return enc_outputs[1]

    def forward(self, X, state):
        # 输入 'X' 的形状：(batch_size, num_steps)
        # 嵌入并将时间步维度放到最前：(num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播 context，使其具有与 X 相同的 num_steps。
        # state[-1] 是编码器最上一层的隐状态，形状为：(batch_size, num_hiddens)
        # repeat 之后形状为：(num_steps, batch_size, num_hiddens)
        context = state[-1].repeat(X.shape[0], 1, 1)
        # 拼接嵌入和上下文变量，作为 GRU 的输入
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        # 变换输出的维度回批量大小在前：(batch_size, num_steps, vocab_size)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size, num_steps, vocab_size)
        # state的形状:(num_layers, batch_size, num_hiddens)
        return output, state


# =====================================================================
# 第二部分：损失函数与遮蔽机制
# =====================================================================

#@save
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项（通常是填充词元）"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size, num_steps, vocab_size)
    # label的形状：(batch_size, num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        # PyTorch 的 CrossEntropyLoss 期望输入预测为 (N, C, d_1, d_2, ..., d_K)
        # 故对 pred 进行 permute，变为 (batch_size, vocab_size, num_steps)
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        # 对每一个样本计算经过屏蔽掩码后的平均损失
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


# =====================================================================
# 第三部分：模型训练与预测逻辑
# =====================================================================

#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # 创建解码器的初始输入：<bos> 引导符
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            # 强制教学 (Teacher Forcing)：使用真实的输出序列作为解码器的输入
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            # 前向传播，net 为 d2l.EncoderDecoder
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
            
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')


#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将 net 设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴，转换成张量形状：(1, num_steps)
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴，初始输入为 '<bos>'，形状：(1, 1)
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元 '<eos>' 被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


# =====================================================================
# 第四部分：预测评估指标 BLEU
# =====================================================================

def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


# =====================================================================
# 第五部分：主程序执行与测试
# =====================================================================

if __name__ == '__main__':
    # 1. 简单对编码器和解码器进行基本健全性检查
    print("------------------------------------------")
    print("进行基础健全性检查 (Sanity Check)...")
    encoder_temp = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    encoder_temp.eval()
    X_temp = torch.zeros((4, 7), dtype=torch.long)
    output_temp, state_temp = encoder_temp(X_temp)
    print("编码器输出形状 (num_steps, batch_size, num_hiddens):", output_temp.shape)
    print("编码器隐状态形状 (num_layers, batch_size, num_hiddens):", state_temp.shape)

    decoder_temp = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    decoder_temp.eval()
    state_init = decoder_temp.init_state(encoder_temp(X_temp))
    output_dec, state_dec = decoder_temp(X_temp, state_init)
    print("解码器输出形状 (batch_size, num_steps, vocab_size):", output_dec.shape)
    print("解码器隐状态形状 (num_layers, batch_size, num_hiddens):", state_dec.shape)

    # 2. 检查 sequence_mask 和 损失函数
    X_mask = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print("测试 sequence_mask (有效长度为 1 和 2):")
    print(sequence_mask(X_mask, torch.tensor([1, 2])))
    
    loss_temp = MaskedSoftmaxCELoss()
    loss_val = loss_temp(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long), torch.tensor([4, 2, 0]))
    print("测试 MaskedSoftmaxCELoss (有效长度分别为 4, 2, 0):")
    print(loss_val)
    print("------------------------------------------")

    # 3. 完整的机器翻译模型训练与测试
    print("加载英-法数据集并开始训练...")
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

    # 加载机器翻译的数据集
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    
    # 初始化完整的 Seq2Seq 编码器-解码器模型
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    
    # 训练模型
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
    
    # 4. 模型预测与翻译评估 (BLEU)
    print("\n进行翻译测试并评估 BLEU 分数:")
    engs = ['go .', "i lost .", "he's calm .", "i'm home ."]
    fras = ['va !', "j'ai perdu .", "il est calme .", "je suis chez moi ."]
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')

    # 在非交互模式下，如果绘制了 Animator 图像，可用 plt.show() 把它显示出来
    plt.show()
