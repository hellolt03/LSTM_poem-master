from main import *
from model import *
from config import *
import torch as t
from generate import *


def userTest():
    print("正在初始化......")
    datas = np.load("tang.npz", allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    model = PoetryModel(len(ix2word), Config.embedding_dim, Config.hidden_dim)
    model.load_state_dict(t.load(Config.model_path, 'cpu'))
    if Config.use_gpu:
        model.to(t.device('cuda'))
    print("初始化完成！\n")

    while True:
        print("欢迎使用唐诗生成器，\n"
              "输入1 进入藏头诗生成模式\n"
              "输入0 退出程序")

        try:
            mode = int(input())
            if mode == 0:
                print("感谢使用唐诗生成器，再见！")
                break
            elif mode == 1:
                print("请输入您想要的诗歌藏头部分，不超过16个字，最好是偶数")
                start_words = str(input())
                gen_poetry = ''.join(gen_acrostic(model, start_words, ix2word, word2ix))
                print("生成的诗句如下：\n%s\n" % (gen_poetry))
            else:
                print("无效的选项，请重新输入！")
        except ValueError:
            print("请输入一个有效的数字！")

if __name__ == '__main__':
    userTest()
