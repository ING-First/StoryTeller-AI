# Fairy Tale Evaluation LoRA Model

## 모델 정보
- **베이스 모델**: skt/A.X-4.0-Light
- **태스크**: 동화 평가 (8개 기준, 1-5점 척도)
- **언어**: 한국어

## 사용법

### 1. 모델 로드
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 베이스 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    "skt/A.X-4.0-Light",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(base_model, "./fairy-tale-model-package")

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("./fairy-tale-model-package", trust_remote_code=True)
```

### 2. 평가 수행
```python
def evaluate_story(story_text):
    prompt = [
        {"role": "system", "content": "당신은 동화를 평가하는 AI입니다. 다음 동화에 대해 아래 8가지 기준에 따라 각각 1~5점으로 점수만 매겨주세요."},
        {"role": "user", "content": f"이 동화를 평가해줘\n\n### 동화:\n{story_text}"}
    ]
    
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    
    result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return result
```

## 평가 기준
1. 시스템 프롬프트 반영 여부
2. 사용자 프롬프트 1 반영 여부
3. 사용자 프롬프트 2 반영 여부
4. 교훈
5. 문법성
6. 비속어 등 부적절한 언어 포함되지 않는지 여부
7. 서사 전개 논리성
8. 동일한 문장 반복되지 않는지 여부

## 파일 구조
- `adapter_config.json`: LoRA 설정
- `adapter_model.safetensors`: LoRA 가중치
- `tokenizer.json`, `tokenizer_config.json`: 토크나이저 파일들
- `model_info.json`: 모델 정보
- `README.md`: 사용법 가이드
