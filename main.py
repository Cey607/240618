import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: 数据加载
data_path = 'data.xlsx'
df = pd.read_excel(data_path)
print(df.head())

# Step 2: 数据预处理
df = df[['bjnr', 'bjlbmc', 'bjlxmc']]
df.columns = ['报警内容', '警情粗类', '警情细类']
df.dropna(subset=['报警内容', '警情粗类', '警情细类'], inplace=True)

# 正则表达式提取报警信息
def extract_alarm_text(text):
    match = re.search(r'报警：(.*)', text)
    if match:
        return match.group(1).strip()
    else:
        return text.strip()

df['报警内容'] = df['报警内容'].apply(extract_alarm_text)

label_encoder_coarse = LabelEncoder()
df['警情粗类编码'] = label_encoder_coarse.fit_transform(df['警情粗类'])

label_encoder_fine = LabelEncoder()
df['警情细类编码'] = label_encoder_fine.fit_transform(df['警情细类'])

import pickle

# 保存 LabelEncoder 对象
with open('label_encoder_coarse.pkl', 'wb') as f:
    pickle.dump(label_encoder_coarse, f)
with open('label_encoder_fine.pkl', 'wb') as f:
    pickle.dump(label_encoder_fine, f)

# Step 3: 文本特征提取
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

X_train_coarse, X_test_coarse, y_train_coarse, y_test_coarse = train_test_split(df['报警内容'], df['警情粗类编码'], test_size=0.2, random_state=205392)
X_train_fine, X_test_fine, y_train_fine, y_test_fine = train_test_split(df['报警内容'], df['警情细类编码'], test_size=0.2, random_state=205392)

def encode_data(texts, labels, max_length=128):
    inputs = tokenizer(texts.tolist(), max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    inputs['labels'] = torch.tensor(labels.values)
    return inputs

train_coarse_encodings = encode_data(X_train_coarse, y_train_coarse)
test_coarse_encodings = encode_data(X_test_coarse, y_test_coarse)
train_fine_encodings = encode_data(X_train_fine, y_train_fine)
test_fine_encodings = encode_data(X_test_fine, y_test_fine)

# Step 4: 模型训练
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: tensor[idx].to(device) for key, tensor in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

train_coarse_dataset = Dataset(train_coarse_encodings)
test_coarse_dataset = Dataset(test_coarse_encodings)
train_fine_dataset = Dataset(train_fine_encodings)
test_fine_dataset = Dataset(test_fine_encodings)

# 定义训练参数和保存检查点的参数
training_args_coarse = TrainingArguments(
    output_dir='./results_coarse',
    num_train_epochs=10,  # 10轮训练
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    logging_dir='./logs_coarse',  # 添加日志目录
    logging_steps=100,  # 每100个步骤记录一次日志
    logging_first_step=True,
    save_steps=500,  # 每500个步骤保存一次检查点
    learning_rate=4e-5,  # 设置学习率
)

# 定义Trainer对象来训练粗类分类模型
model_coarse = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder_coarse.classes_))
model_coarse.to(device)

trainer_coarse = Trainer(
    model=model_coarse,
    args=training_args_coarse,
    train_dataset=train_coarse_dataset,
    eval_dataset=test_coarse_dataset,
    tokenizer=tokenizer,
)

# 训练粗类分类模型
trainer_coarse.train()

# 保存粗类分类模型的最终检查点
trainer_coarse.save_model('./checkpoint/coarse_model')

# 定义细类分类模型的训练参数
training_args_fine = TrainingArguments(
    output_dir='./results_fine',
    num_train_epochs=20, 
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    logging_dir='./logs_fine',  # 添加日志目录
    logging_steps=100,  # 每100个步骤记录一次日志
    logging_first_step=True,
    save_steps=500,  # 每500个步骤保存一次检查点
    learning_rate=5e-5,  # 设置学习率
)

# 定义Trainer对象来训练细类分类模型
model_fine = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder_fine.classes_))
model_fine.to(device)

trainer_fine = Trainer(
    model=model_fine,
    args=training_args_fine,
    train_dataset=train_fine_dataset,
    eval_dataset=test_fine_dataset,
    tokenizer=tokenizer,
)

# 训练细类分类模型
trainer_fine.train()

# 保存细类分类模型的最终检查点
trainer_fine.save_model('./checkpoint/fine_model')

# Step 5: 模型评估
preds_coarse = trainer_coarse.predict(test_coarse_dataset)
y_pred_coarse = preds_coarse.predictions.argmax(-1)
print("粗类分类报告:\n", classification_report(y_test_coarse, y_pred_coarse, labels=range(len(label_encoder_coarse.classes_)), target_names=label_encoder_coarse.classes_))

preds_fine = trainer_fine.predict(test_fine_dataset)
y_pred_fine = preds_fine.predictions.argmax(-1)
print("细类分类报告:\n", classification_report(y_test_fine, y_pred_fine, labels=range(len(label_encoder_fine.classes_)), target_names=label_encoder_fine.classes_))

# Step 6: 系统设计与实现
def classify_alarm(alarm_text, model_coarse, model_fine):
    inputs = tokenizer(alarm_text, return_tensors='pt', max_length=128, truncation=True, padding='max_length').to(device)
    
    coarse_outputs = model_coarse(**inputs)
    coarse_prediction_encoded = coarse_outputs.logits.argmax(-1).item()
    coarse_prediction = label_encoder_coarse.inverse_transform([coarse_prediction_encoded])[0]

    fine_outputs = model_fine(**inputs)
    fine_prediction_encoded = fine_outputs.logits.argmax(-1).item()
    fine_prediction = label_encoder_fine.inverse_transform([fine_prediction_encoded])[0]

    return coarse_prediction, fine_prediction

# 示例程序：接收用户输入并分类
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
