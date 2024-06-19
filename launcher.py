from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pickle
import os

app = Flask(__name__)
# 设置模板文件夹路径
template_dir = os.path.abspath('./templates')
app.template_folder = template_dir

# 加载模型和标记器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('./checkpoint/coarse_model')
model_coarse = BertForSequenceClassification.from_pretrained('./checkpoint/coarse_model')
model_fine = BertForSequenceClassification.from_pretrained('./checkpoint/fine_model')

# 加载LabelEncoder
with open('label_encoder_coarse.pkl', 'rb') as f:
    label_encoder_coarse = pickle.load(f)
with open('label_encoder_fine.pkl', 'rb') as f:
    label_encoder_fine = pickle.load(f)

# 将模型移动到设备
model_coarse.to(device)
model_fine.to(device)

# 定义分类函数
def classify_alarm(alarm_text):
    inputs = tokenizer(alarm_text, return_tensors='pt', max_length=128, truncation=True, padding='max_length').to(device)
    
    coarse_outputs = model_coarse(**inputs)
    coarse_prediction_encoded = coarse_outputs.logits.argmax(-1).item()
    coarse_prediction = label_encoder_coarse.inverse_transform([coarse_prediction_encoded])[0]

    fine_outputs = model_fine(**inputs)
    fine_prediction_encoded = fine_outputs.logits.argmax(-1).item()
    fine_prediction = label_encoder_fine.inverse_transform([fine_prediction_encoded])[0]

    return coarse_prediction, fine_prediction

# 定义路由和处理函数
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_text():
    if request.method == 'POST':
        data = request.form
        if 'text' not in data:
            return jsonify({'error': 'Missing text parameter'})
        
        alarm_text = data['text']
        if not alarm_text:
            return jsonify({'error': 'Empty text input'})

        coarse_result, fine_result = classify_alarm(alarm_text)
        return jsonify({
            'coarse_prediction': coarse_result,
            'fine_prediction': fine_result
        })
    else:
        return jsonify({'error': 'Method Not Allowed'})

if __name__ == '__main__':
    app.run(debug=True)
