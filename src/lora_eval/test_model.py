#!/usr/bin/env python3
"""
파인튜닝된 동화 평가 모델 테스트 스크립트
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model():
    """모델과 토크나이저 로드"""
    print("Loading model...")
    
    # 베이스 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        "skt/A.X-4.0-Light",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(base_model, ".")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(".", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def evaluate_fairy_tale(model, tokenizer, story):
    """동화 평가"""
    prompt = [
        {
            "role": "system", 
            "content": "당신은 동화를 평가하는 AI입니다. 다음 동화에 대해 아래 8가지 기준에 따라 각각 1~5점으로 점수만 매겨주세요."
        },
        {
            "role": "user", 
            "content": f"이 동화를 평가해줘\n\n### 동화:\n{story}"
        }
    ]
    
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    
    if torch.cuda.is_available():
        inputs = inputs.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return result

if __name__ == "__main__":
    # 모델 로드
    model, tokenizer = load_model()
    
    # 테스트 동화
    test_story = """작은 마을에는 겨울이 오면 모든 것이 얼음으로 덮였어요. 아이들은 하얀 세상에서 눈싸움을 하며 행복한 시간을 보냈어요. 그런데 어느 날, 바람이 이상하게도 귀엽게 휙 불어오더니 작은 마법 양탄자를 가져왔어요."""
    
    # 평가 수행
    print("🧚‍♀️ 동화 평가 테스트")
    print(f"동화: {test_story}")
    print("\n평가 결과:")
    
    result = evaluate_fairy_tale(model, tokenizer, test_story)
    print(result)
