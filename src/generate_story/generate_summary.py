from datetime import date
import re
from typing import List, Optional
from .lora_manager import get_lora_manager, ensure_model_loaded, switch_to_lora
import torch

class Summarizer:
    def __init__(self):
        self.lora_manager = get_lora_manager()
        self._ensure_model_loaded()
    
    def _ensure_model_loaded(self):
        if not ensure_model_loaded():
            raise RuntimeError("베이스 모델 로딩 실패")
        
        if not switch_to_lora("summary"):
            print("[Summarizer] WARNING: summary LoRA 로딩 실패, 베이스 모델 사용")
    
    @property
    def model(self):
        return self.lora_manager.get_current_model()
    
    @property
    def tokenizer(self):
        return self.lora_manager.get_tokenizer()
    
    @property
    def device(self):
        return self.lora_manager.device

    def generate_summary(
        self,
        uid: int,
        type: int,
        title: str,
        contents: str,
        max_new_tokens: int = 80,
    ) -> dict:
        
        # summary LoRA 활성화 확인
        if self.lora_manager.get_current_lora_name() != "summary":
            switch_to_lora("summary")

        # 시스템 프롬프트
        system_prompt = (
            "System: 당신은 간결하고 핵심적인 요약을 작성하는 전문가입니다. "
            "출력은 반드시 2문장 이내로 작성하고, 문장은 자연스럽고 문법적으로 올바르게 구성하세요."
            "동화의 교훈 위주로 핵심만 요약하세요."
        )

        prompt = (
            f"{system_prompt}\n\n"
            "User: 다음 글을 2문장 이내로 핵심만 간단하게 요약해 주세요.:\n"
            f"{contents}\n\nAssistant:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True,
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    use_cache=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        except Exception as e:
            return {
                "uid": uid,
                "type": type,
                "title": title,
                "summary": f"[ERROR] {e}",
                "contents": contents,
                "createDate": date.today(),
                "success": False,
            }

        # 디코딩 및 후처리
        gen_tokens = outputs[0][input_len:]
        summary = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        for stop in ["User:", "Assistant:", "USER:", "ASSISTANT:"]:
            if stop in summary:
                summary = summary.split(stop)[0]
                break
            
        summary = re.sub(r"[A-Za-z0-9.,:;!?\"'()\[\]{}<>@#$%^&*+=/_\-]+", "", summary)
        
        return {
            "uid": uid,
            "type": type,
            "title": title,
            "summary": summary.strip(),
            "contents": contents,
            "create_date": date.today(),
            "success": True,
        }

    def generate_page_summaries(self, contents: List[str], max_new_tokens: int = 77) -> List[str]:
        # summary LoRA 활성화 확인
        if self.lora_manager.get_current_lora_name() != "summary":
            switch_to_lora("summary")
            
        results = []

        for chunk in contents:
            prompt = (
                "System: 당신은 간결하고 핵심적인 요약을 작성하는 전문가입니다. "
                "출력은 반드시 1문장으로 작성하세요.\n\n"
                f"User: 다음 글을 1문장으로 요약해 주세요:\n{chunk}\n\nAssistant:"
            )

            inputs = self.tokenizer(
                prompt, return_tensors="pt", max_length=1024, truncation=True, padding=True
            )
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=77,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            gen_tokens = outputs[0][input_len:]
            summary = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            for stop in ["User:", "Assistant:", "USER:", "ASSISTANT:"]:
                if stop in summary:
                    summary = summary.split(stop)[0].strip()
                    break
                
            summary = re.sub(r"[A-Za-z0-9]+", "", summary)
            results.append(summary)

        return results