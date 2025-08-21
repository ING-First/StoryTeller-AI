# LoRA Fine-tuned A.X-4.0 Model (Best Checkpoint)

## 모델 정보
- Base Model: skt/A.X-4.0-Light
- Fine-tuning Method: LoRA (Low-Rank Adaptation)
- Best Checkpoint: Step 1400
- Training Loss: 1.972
- Date: 1754977631.13561

## 사용 방법
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 베이스 모델 로드
base_model = AutoModelForCausalLM.from_pretrained("skt/A.X-4.0-Light")
tokenizer = AutoTokenizer.from_pretrained("skt/A.X-4.0-Light")

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(base_model, "./lora-ax40-best-1400")

# 추론
prompt = "User: 안녕하세요!\n\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 모델 성능
- Training Loss at Step 1400: 1.972
- LoRA Parameters: ~10M
- Total Parameters: ~7.2B (only 0.14% trainable)
