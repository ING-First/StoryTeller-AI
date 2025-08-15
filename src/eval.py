from __future__ import annotations
from typing import List, Dict, Optional, Any
import time
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline
from peft import PeftModel
class StoryEvaluator:
    CRITERIA = [
        "시스템 프롬프트 반영 여부",
        "사용자 프롬프트 1 반영 여부",
        "사용자 프롬프트 2 반영 여부",
        "교훈",
        "문법성",
        "비속어 등 부적절한 언어 포함되지 않는지 여부",
        "서사 전개 논리성",
        "동일한 문장 반복되지 않는지 여부",
    ]

    _LINE_RE = re.compile(
        r'(?m)^\s*(?:[-•]\s*)?(?:\d+\s*[\.\)]\s*)?[^\n:]+[:：]\s*([1-5])\s*점'
    )

    def __init__(
        self,
        base_model_name: str = "skt/A.X-4.0-Light",
        lora_model_path: str = "./lora_eval",
        max_new_tokens: int = 100,
        repetition_penalty: float = 1.1,
        do_sample: bool = False,
        device_map: Optional[str] = None,
    ):
        self.base_model_name = base_model_name
        self.lora_model_path = lora_model_path
        self.evaluation_criteria = self.CRITERIA
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.torch_dtype = (
            torch.bfloat16
            if torch.cuda.is_available()
            and torch.cuda.get_device_capability(0)[0] >= 8
            else torch.float16
        )
        if device_map is None:
            device_map = "cuda:0" if torch.cuda.is_available() else "auto"
        self.device_map = device_map

        # Base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        try:
            self.model = PeftModel.from_pretrained(
                self.base_model,
                self.lora_model_path,
                torch_dtype=self.torch_dtype,
            )
        except Exception as e:
            print(f"[WARN] LoRA 모델 로드 실패. 베이스 모델 로드: {e}")
            self.model = self.base_model

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        gen_cfg = GenerationConfig(
            do_sample=self.do_sample,
            temperature=None if not self.do_sample else 0.7,
            top_p=None if not self.do_sample else 0.9,
            top_k=None if not self.do_sample else 50,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
        )
        self.model.generation_config = gen_cfg

        # Pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            batch_size=1,
            device_map=None,
        )

    @staticmethod
    def _system_prompt() -> str:
        return (
            "당신은 동화를 평가하는 AI입니다. 다음 동화에 대해 아래 8가지 기준에 따라 "
            "각각 1~5점으로 점수만 매겨주세요. 이유는 생략하고 점수만 출력하세요."
        )

    def make_chat_prompt(self, story_text: str) -> List[Dict[str, str]]:
        crit = self.evaluation_criteria
        crit_lines = "\n".join([f"{i+1}. {c}" for i, c in enumerate(crit)])
        answer_fmt = "\n".join([f"{i+1}. {c}: X점" for i, c in enumerate(crit)])

        return [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": (
                    f"이 동화를 평가해줘\n\n### 동화:\n{story_text}\n\n"
                    f"### 평가 기준:\n{crit_lines}\n\n"
                    f"### 답변 형식:\n{answer_fmt}"
                ),
            },
        ]

    def evaluate_single_story_fast(
        self,
        story_text: str,
        index: int = 0,
        parse_scores_only: bool = False,
        expected_items: int = 8,
    ) -> Dict[str, Any]:
        
        chat = self.make_chat_prompt(story_text)

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

        t0 = time.time()
        with torch.inference_mode():
            try:
                out = self.pipe(
                    prompt_text,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=None if not self.do_sample else 0.7,
                    repetition_penalty=self.repetition_penalty,
                    use_cache=True,
                )[0]["generated_text"]
            except Exception as e:
                out = f"평가 생성 실패: {e}"
        t1 = time.time()

        gpu_mb = 0
        if torch.cuda.is_available():
            gpu_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024))

        result: Dict[str, Any] = {
            "index": index,
            "story": story_text,
            "evaluation": out,
            "evaluation_time_sec": round(t1 - t0, 4),
            "gpu_memory_used_mb": gpu_mb,
        }
        if parse_scores_only:
            result["scores"] = self.parse_scores_only(out, expected=expected_items)
        return result

    @classmethod
    def parse_scores_only(cls, text: str, expected: int = 8) -> List[int]:
        scores = [int(s) for s in cls._LINE_RE.findall(text)]
        if len(scores) < expected:
            fallback = re.findall(r'([1-5])\s*점', text)
            scores = [int(s) for s in fallback][:expected]
        return scores[:expected]