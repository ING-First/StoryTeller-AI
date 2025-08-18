from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from peft import PeftModel
from datetime import date
import torch
import re


class Summarizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        BASE_DIR = Path(__file__).resolve().parent.parent
        self.lora_path = BASE_DIR.parent / "models" / "lora_summary"

        print(f"Device: {self.device}, dtype: {self.dtype}")

        self.model = None
        self.tokenizer = None

    def load_lora_model(
        self,
        base_model_id="skt/A.X-4.0-Light",
    ):
        print(f"LoRA 모델 로딩 중...")
        print(f"베이스 모델: {base_model_id}")
        print(f"LoRA 어댑터: {self.lora_path}")

        try:
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_id, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.tokenizer.padding_side = "left"

            # 베이스 모델 로드
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                device_map='auto' if self.device == 'cuda' else None,
                torch_dtype=self.dtype,
                trust_remote_code=True
            )

            # LoRA 어댑터 적용
            self.model = PeftModel.from_pretrained(base_model, self.lora_path)
            if self.device == "cpu":
                self.model.to(self.device)

            self.model.eval()
            print("LoRA 모델 로딩 완료")
            return True

        except Exception as e:
            print(f"LoRA 모델 로딩 실패: {str(e)}")
            return False

    def generate_summary(
        self,
        uid: int,
        type: int,
        title: str,
        contents: str,
        max_new_tokens: int = 200,
    ) -> dict:

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load_lora_model()을 먼저 호출하세요.")

        # 시스템 프롬프트 추가
        system_prompt = (
            "System: 당신은 간결하고 핵심적인 요약을 작성하는 전문가입니다. "
            "출력은 반드시 2문장 이내로 작성하고, 문장은 자연스럽고 문법적으로 올바르게 구성하세요."
            "동화의 교훈 위주로 핵심만 요약하세요."
        )

        # prompt 설정
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

        if self.device == "cuda":
            torch.cuda.synchronize()

        try:
            # 생성 설정
            with torch.inference_mode():
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

        if self.device == "cuda":
            torch.cuda.synchronize()

        # 디코딩
        gen_tokens = outputs[0][input_len:]
        summary = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # 후처리
        for stop in ["User:", "Assistant:", "USER:", "ASSISTANT:"]:
            if stop in summary:
                summary = summary.split(stop)[0]
                break

        return {
            "uid": uid,
            "type": type,
            "title": title,
            "summary": summary.strip(),
            "contents": contents,
            "create_date": date.today(),
            "success": True,
        }

    # 2문장 단위로 분할
    def split_into_chunks(self, contents: str, sentences_per_chunk: int = 2) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', contents.strip())
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = " ".join(sentences[i:i+sentences_per_chunk])
            if chunk:
                chunks.append(chunk)
        return chunks

    # 페이지별 요약
    def generate_page_summaries(self, contents: str, max_new_tokens: int = 77) -> list[str]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load_lora_model()을 먼저 호출하세요.")

        chunks = self.split_into_chunks(contents, sentences_per_chunk=2)
        results = []

        for i, chunk in enumerate(chunks):
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

            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
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

            results.append(summary)

        return results