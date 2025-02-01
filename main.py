import torch as t
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from model import *
from torchnet import meter
import tqdm
import matplotlib.pyplot as plt
import os
import jieba
from nltk.translate.bleu_score import sentence_bleu
from config import *
from test import *

# 确保已经下载了nltk的必要数据
import nltk
nltk.download('punkt')

def calculate_perplexity(loss):
    # 困惑度 = exp(损失)
    return t.exp(t.tensor(loss))

def calculate_bleu(predictions, references):
    """
    计算 BLEU 分数，predictions 为生成的文本列表，references 为参考文本列表。
    :param predictions: 生成的文本 (list of list of str)
    :param references: 参考文本 (list of list of str)
    :return: BLEU分数
    """
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        # 取出生成的句子和参考句子（假设它们是列表，取第一个元素）
        pred_text = str(pred[0])  # 生成的诗句，确保是字符串类型
        ref_text = str(ref[0])    # 参考的诗句，确保是字符串类型

        # 对生成的句子和参考句子进行中文分词
        pred_tokens = list(jieba.cut(pred_text))  # 生成的句子
        ref_tokens = list(jieba.cut(ref_text))    # 参考句子
        
        bleu_scores.append(sentence_bleu([ref_tokens], pred_tokens))  # 计算 BLEU 分数
    return np.mean(bleu_scores)

def train():
    if Config.use_gpu:
        Config.device = t.device("cuda")
    else:
        Config.device = t.device("cpu")
    device = Config.device
    # 获取数据
    datas = np.load("tang.npz", allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    data = t.from_numpy(data)
    dataloader = DataLoader(data,
                            batch_size=Config.batch_size,
                            shuffle=True,
                            num_workers=2)

    # 定义模型
    model = PoetryModel(len(word2ix),
                        embedding_dim=Config.embedding_dim,
                        hidden_dim=Config.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    criterion = nn.CrossEntropyLoss()
    if Config.model_path:
        model.load_state_dict(t.load(Config.model_path, map_location='cpu'))
    # 转移到相应计算设备上
    model.to(device)
    loss_meter = meter.AverageValueMeter()

    # 可视化设置
    epoch_losses = []  # 用于存储每个epoch的损失值
    epoch_perplexities = []  # 用于存储每个epoch的困惑度
    epoch_bleu_scores = []  # 用于存储每个epoch的BLEU分数

    # 进行训练
    f = open('result.txt', 'w', encoding='gbk', errors='replace')
    for epoch in range(Config.epoch):
        loss_meter.reset()
        all_predictions = []  # 用于保存生成的文本
        all_references = []   # 用于保存参考文本
        for li, data_ in tqdm.tqdm(enumerate(dataloader)):
            data_ = data_.long().transpose(1, 0).contiguous().to(device)
            optimizer.zero_grad()
            # n个句子，前n-1句作为输入，后n-1句作为输出，二者一一对应
            input_, target = data_[:-1, :], data_[1:, :]
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())

            # 进行可视化
            if (1 + li) % Config.plot_every == 0:
                print(f"训练损失为 {loss_meter.mean:.4f}")
                f.write(f"训练损失为 {loss_meter.mean:.4f}\n")
                for word in list(u"春江花朝秋月夜"):
                    gen_poetry = ''.join(generate(model, word, ix2word, word2ix))
                    print(gen_poetry)
                    f.write(gen_poetry + "\n\n")
                    f.flush()

                # 保存生成的文本和参考文本
                all_predictions.append([gen_poetry])  # 生成的诗句
                all_references.append([list(u"春江花朝秋月夜")])  # 参考文本，可以是某个实际的诗句或真实标签

        # 保存每个epoch的损失，用于后续绘图
        epoch_losses.append(loss_meter.mean)
        
        # 计算困惑度
        perplexity = calculate_perplexity(loss_meter.mean)
        epoch_perplexities.append(perplexity.item())

        # 计算 BLEU 分数
        bleu_score = calculate_bleu(all_predictions, all_references)
        epoch_bleu_scores.append(bleu_score)

        # 打印并保存评估指标
        print(f"Epoch {epoch} Perplexity: {perplexity:.4f}, BLEU Score: {bleu_score:.4f}")

        # 保存模型
        t.save(model.state_dict(), f'{Config.model_prefix}_{epoch}.pth')

    # 绘制训练损失曲线并保存
    os.makedirs('outputs', exist_ok=True)
    plt.plot(range(Config.epoch), epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/training_loss_curve.png')  # 保存图像
    plt.show()  # 显示图像

    # 绘制 Perplexity 和 BLEU 曲线
    plt.figure()
    plt.plot(range(Config.epoch), epoch_perplexities, label='Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Perplexity per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/perplexity_curve.png')
    plt.show()

    plt.figure()
    plt.plot(range(Config.epoch), epoch_bleu_scores, label='BLEU Score')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/bleu_score_curve.png')
    plt.show()

if __name__ == '__main__':
    train()
