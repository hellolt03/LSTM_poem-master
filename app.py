from flask import Flask, render_template, request, jsonify
import torch as t
import numpy as np
from generate import gen_acrostic
from config import Config
from model import PoetryModel

app = Flask(__name__)

# 加载模型
datas = np.load("tang.npz", allow_pickle=True)
ix2word = datas['ix2word'].item()
word2ix = datas['word2ix'].item()
model = PoetryModel(len(ix2word), Config.embedding_dim, Config.hidden_dim)
model.load_state_dict(t.load(Config.model_path, 'cpu'))
if Config.use_gpu:
    model.to(t.device('cuda'))

@app.route('/')
def index():
    # 返回前端的HTML页面
    return render_template('index.html')

@app.route('/generate_poetry', methods=['POST'])
def generate_poetry():
    data = request.json
    start_words = data.get('start_words')

    # 调用生成诗句的函数
    poetry = ''.join(gen_acrostic(model, start_words, ix2word, word2ix))

    return jsonify({
        'success': True,
        'poetry': poetry
    })

if __name__ == '__main__':
    app.run(debug=True)
