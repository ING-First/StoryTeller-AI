from __future__ import annotations
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline
from peft import PeftModel
import time
import torch
import re
import json

class Evaluator:
    def __init__(
        self,
        base_model_id: str = "skt/A.X-4.0-Light",
        lora_path: str = "./lora-ax40-eval",
        device_map: Optional[str] = None,
        offload_folder: Optional[str] = "./offload",
        max_new_tokens: int = 100,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ):
        self.base_model_id = base_model_id
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if device_map is None:
            device_map = "auto" if torch.cuda.is_available() else None
        self.device_map = device_map
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch.cuda.is_available():
            cc_major = torch.cuda.get_device_capability(0)[0]
            dtype = torch.bfloat16 if cc_major >= 8 else torch.float16
        else:
            dtype = torch.float32

        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map=self.device_map,
            offload_folder=offload_folder,
            low_cpu_mem_usage=False,
            offload_state_dict=False,
            trust_remote_code=True,
        )
        
        self.model = PeftModel.from_pretrained(base, lora_path)
        if self.device == "cpu":
            self.model.to(self.device)
        self.model.eval()

        gen_cfg = GenerationConfig(
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
        )
        self.model.generation_config = gen_cfg

        self.chat_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
        )

        # 평가 기준
        self.evaluation_criteria: List[str] = [
            "시스템 프롬프트 반영 여부",
            "사용자 프롬프트 1 반영 여부",
            "사용자 프롬프트 2 반영 여부",
            "교훈",
            "문법성",
            "비속어 등 부적절한 언어 포함되지 않는지 여부",
            "서사 전개 논리성",
            "동일한 문장 반복되지 않는지 여부",
        ]

    # 프롬프트 빌더
    def make_chat_prompt(self, story_text: str) -> list[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "당신은 동화를 평가하는 AI입니다. 다음 동화에 대해 아래 8가지 기준에 따라 "
                    "각각 1~5점으로 점수만 입력해주세요. (이유 제외)"
                ),
            },
            {
                "role": "user",
                "content": f"""### 동화:
        {story_text}

        ### 평가 기준:
        1. {self.evaluation_criteria[0]}
        2. {self.evaluation_criteria[1]}
        3. {self.evaluation_criteria[2]}
        4. {self.evaluation_criteria[3]}
        5. {self.evaluation_criteria[4]}
        6. {self.evaluation_criteria[5]}
        7. {self.evaluation_criteria[6]}
        8. {self.evaluation_criteria[7]}
        """,
            },
        ]

    def evaluate_text(
        self,
        story_text: str,
        max_new_tokens: Optional[int] = None,
        deterministic: bool = False,
        return_metrics: bool = True,
    ) -> Dict[str, Any]:
        chat_input = self.make_chat_prompt(story_text)
        prompt_text = self.tokenizer.apply_chat_template(
            chat_input, tokenize=False, add_generation_prompt=True
        )

        pipe_kwargs = {}
        if deterministic:
            pipe_kwargs["do_sample"] = False

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        out = self.chat_pipe(
            prompt_text,
            max_new_tokens=max_new_tokens or self.model.generation_config.max_new_tokens,
            **pipe_kwargs
        )[0]["generated_text"]
        t1 = time.time()

        # 점수 추출
        scores = re.findall(r"\b([1-5])\b", out)
        scores = list(map(int, scores[:8]))

        metrics = {}
        if return_metrics:
            eval_time = round(t1 - t0, 3)
            gpu_mb = 0.0
            if torch.cuda.is_available():
                gpu_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
            metrics = {"evaluation_time_sec": eval_time, "gpu_memory_used_mb": gpu_mb}

        return {
            "story": story_text,
            "chat_input": chat_input,
            "evaluation_raw": out,
            "scores": scores,
            **metrics,
        }