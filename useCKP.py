import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载标记器
tokenizer = BertTokenizer.from_pretrained('./checkpoint/coarse_model')

# 加载模型
model_coarse = BertForSequenceClassification.from_pretrained('./checkpoint/coarse_model')
model_fine = BertForSequenceClassification.from_pretrained('./checkpoint/fine_model')

# 加载LabelEncoder
with open('label_encoder_coarse.pkl', 'rb') as f:
    label_encoder_coarse = pickle.load(f)
with open('label_encoder_fine.pkl', 'rb') as f:
    label_encoder_fine = pickle.load(f)

# 移动模型到设备
model_coarse.to(device)
model_fine.to(device)

# 定义分类函数
def classify_alarm(alarm_text, model_coarse, model_fine):
    inputs = tokenizer(alarm_text, return_tensors='pt', max_length=128, truncation=True, padding='max_length').to(device)
    
    coarse_outputs = model_coarse(**inputs)
    coarse_prediction_encoded = coarse_outputs.logits.argmax(-1).item()
    coarse_prediction = label_encoder_coarse.inverse_transform([coarse_prediction_encoded])[0]

    fine_outputs = model_fine(**inputs)
    fine_prediction_encoded = fine_outputs.logits.argmax(-1).item()
    fine_prediction = label_encoder_fine.inverse_transform([fine_prediction_encoded])[0]

    return coarse_prediction, fine_prediction

# 示例使用
def main():
    print("请输入报警内容（输入exit退出程序）：")
    while True:
        alarm_text = input("> ").strip()
        if alarm_text.lower() == "exit":
            print("程序已退出。")
            break
        coarse_result, fine_result = classify_alarm(alarm_text, model_coarse, model_fine)
        print(f"粗类预测: {coarse_result}")
        print(f"细类预测: {fine_result}")
if __name__ == "__main__":
    main()
