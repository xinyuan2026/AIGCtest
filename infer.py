import torch
from transformers import AutoTokenizer
from whosai_model import WhosAIModel

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "roberta-large"
num_classes = 5
labels = ["Human", "GPT-3.5", "GPT-4", "LLaMA", "Claude"]

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = WhosAIModel(model_name, num_classes).to(device)
model.eval()

# 测试文本（你可以改成自己的内容）
test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "人工智能正在深刻改变我们的生活方式，从日常工作到娱乐场景，无处不在。",
    "今天天气真不错，我打算出去散散步，看看公园里的花。"
]

# 推理
with torch.no_grad():
    for text in test_texts:
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        ).to(device)
        
        logits = model(**inputs)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()
        
        print(f"文本：{text}")
        print(f"预测来源：{labels[pred_idx]}")
        print(f"AI生成概率：{(1 - probs[0].item()) * 100:.2f}%")
        print("-" * 60)