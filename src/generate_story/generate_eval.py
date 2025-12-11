from __future__ import annotations
from typing import List, Dict, Optional, Any
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline
from .lora_manager import get_lora_manager, ensure_model_loaded, switch_to_lora
from peft import PeftModel
import torch
import re
class StoryEvaluator:
    CRITERIA = [
        "시스템 프롬프트 반영 여부(별도로 없는 경우 5점)",
        "교훈",
        "문법성",
        "비속어 등 부적절한 언어 포함되지 않는지 여부",
        "서사 전개 논리성",
        "동일한 문장 반복되지 않는지 여부",
    ]

    _LINE_RE = re.compile(
        r'(?m)^\s*(?:[-•]\s*)?(?:\d+\s*[\.\)]\s*)?[^\n:]+[:：]\s*([1-5])\s*점'
    )

    def __init__(self):
        self.lora_manager = get_lora_manager()        
        self.evaluation_criteria = self.CRITERIA
        self._ensure_model_loaded()
        
    def _ensure_model_loaded(self):
        if not ensure_model_loaded():
            raise RuntimeError("베이스 모델 로딩 실패")
        
        if not switch_to_lora("eval"):
            print("[StoryEvaluator] WARNING: eval LoRA 로딩 실패, 베이스 모델 사용")

    @property
    def model(self):
        return self.lora_manager.get_current_model()

    @property
    def tokenizer(self):
        return self.lora_manager.get_tokenizer()

    @staticmethod
    def _system_prompt() -> str:
        return (
            "당신은 동화를 평가하는 AI입니다. 다음 동화에 대해 아래 6가지 기준에 따라 "
            "각각 1~5점으로 점수를 매기고, 각 항목별로 그렇게 평가한 이유를 간단히 설명해주세요."
        )

    def make_chat_prompt(self, story_text: str, prompt: str = "") -> List[Dict[str, str]]:
        crit = self.evaluation_criteria
        crit_lines = "\n".join([f"{i+1}. {c}" for i, c in enumerate(crit)])
        answer_fmt = "\n".join([f"{i+1}. {c}: X점 (이유: ...)" for i, c in enumerate(crit)])

        return [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": (
                    f"이 동화를 평가해줘\n\n"
                    f"### 동화:\n{story_text}\n\n"
                    f"### (동화 생성 시 사용된) 시스템 프롬프트: {prompt}\n\n"
                    f"### 평가 기준:\n{crit_lines}\n\n"
                    f"### 답변 형식:\n{answer_fmt}"
                ),
            },
        ]

    def evaluate_single_story_fast(
        self,
        story_text: str,
        prompt: str,
        parse_scores_only: bool = False,
        expected_items: int = 6,
        max_new_tokens: int = 300,
        do_sample: bool = False,
        repetition_penalty: float = 1.1,
    ) -> Dict[str, Any]:
        
        # eval LoRA 활성화 확인
        if self.lora_manager.get_current_lora_name() != "eval":
            switch_to_lora("eval")
        
        chat = self.make_chat_prompt(story_text, prompt)

        try:
            prompt_text = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_text = (
                f"[SYSTEM]\n{chat[0]['content']}\n\n[USER]\n{chat[1]['content']}\n\n[ASSISTANT]\n"
            )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # 토크나이징
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True,
        )
        
        if self.lora_manager.device == "cuda":
            inputs = {k: v.to(self.lora_manager.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            try:
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=None if not do_sample else 0.7,
                    repetition_penalty=repetition_penalty,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                gen_tokens = outputs[0][input_len:]
                out = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                
            except Exception as e:
                out = f"평가 생성 실패: {e}"

        result: Dict[str, Any] = {
            "story": story_text,
            "evaluation": out
        }
        result["scores"] = self.parse_scores_only(out, expected=expected_items)
        return result

    @classmethod
    def parse_scores_only(
        cls,
        text: str,
        expected: int = 6,
    ) -> List[int]:
        scores = [0] * expected
        lines = text.strip().splitlines()
        for line in lines:
            # "1. 항목명: 5점 (이유: ...)" 형식에서 점수만 추출
            # 점수 뒤에 오는 괄호나 다른 내용은 무시
            m = re.match(r"^\s*(\d+)\.\s*[^\:：]+[:：]\s*([0-5])\s*점", line)
            if m:
                idx = int(m.group(1)) - 1
                val = m.group(2)
                if val.isdigit() and 0 <= idx < expected:
                    scores[idx] = int(val)
        return scores[:expected]