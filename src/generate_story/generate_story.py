import os, re, time, torch
from typing import Optional, Dict, Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

class StoryBookGenerator:
    def __init__(self,
                 repo_or_path: str = "kkuriyoon/QLoRA-ax4-StoryTeller",
                 hf_token: Optional[str] = None,
                 max_new_tokens: int = 280,
                 temperature: float = 0.8,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.05,
                 length_hint: str = "6~8문장",
                 reading_level: str = "해당 나이 또래가 술술 읽을 수 있는 난이도",
                 style_override: Optional[str] = None):
        
        # model
        self.repo_or_path = repo_or_path
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "").strip() or None

        # generate
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        # prompt
        self.length_hint = length_hint
        self.reading_level = reading_level
        self.style_override = style_override
        self.safety = "폭력/공포/혐오/연령불가 요소 금지"
        self.genre_guides = {
            "동화": "따뜻하고 포근한 톤, 일상적 갈등과 작은 해결, 의성어/의태어 소량",
            "모험": "경쾌한 진행, 목표-장애-성장의 3막, 공간 이동과 작은 퀘스트",
            "미스터리": "부드러운 호기심 유발, 위험 최소화, 단서-추론-해결의 흐름",
            "판타지": "상상력 가득한 세계관, 마법/상징을 은유적으로 사용",
            "SF": "미래/과학 요소를 쉽고 안전하게 설명, 기술은 친근한 도구처럼",
            "일상": "친구/가족/학교/동네 등 공감 포인트 중심의 소소한 사건",
            "동시": "리듬/반복/이미지를 살린 운율, 짧은 행과 명료한 메시지",
        }

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None


    def _select_dtype_and_device_map(self):
        if torch.cuda.is_available():
            return torch.float16, "auto"
        elif torch.backends.mps.is_available():
            return torch.float16, {"": "mps"}
        return torch.float32, {"": "cpu"}

    def _split_title_content(self, text: str) -> Tuple[str, str]:
        s = text.strip()
        titles = re.findall(r"제목\s*:\s*(.+)", s)
        title = titles[0].strip() if titles else "제목 없음"
        title = re.sub(r"[\*\n\\]+", " ", title).strip()
        content = re.sub(r"제목\s*:\s*.+", "", s)
        content = re.sub(r"[\*\n\\#-]+", " ", content).strip()
        return title, content

    def _split_sentences_kor(self, s: str):
        s = re.sub(r"\s+", " ", s.strip())
        pat = re.compile(r'.*?(?:다\.|요\.|[.!?…])')
        sentences = pat.findall(s)
        tail = s[sum(len(x) for x in sentences):].strip()
        if tail: sentences.append(tail)
        return [x.strip() for x in sentences if x.strip()]

    def _split_into_chunks(self, contents: str) -> List[str]:
        """무조건 2문장씩 묶어서 반환"""
        sents = self._split_sentences_kor(contents or "")
        chunks: List[str] = []
        for i in range(0, len(sents), 2):
            chunk = " ".join(sents[i:i+2]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks


    def load(self):
        print("-- 모델 로딩중")
        torch_dtype, device_map = self._select_dtype_and_device_map()
        token_kw = {"token": self.hf_token} if self.hf_token else {}

        tok = AutoTokenizer.from_pretrained(self.repo_or_path, use_fast=True, **token_kw)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.repo_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **token_kw
        )
        self.tokenizer, self.model = tok, model
        print("-- 모델 로딩 완료")

    def _build_prompt(self, name: str, age: int, genre: str) -> str:
        guide = self.style_override or self.genre_guides.get(genre, "장르적 관습을 유아 친화적으로 순화하여 반영")
        
        return f"""
당신은 아동 문학 작가이자 언어발달 선생님입니다. {age}살 아이 '{name}'에게 딱 맞는 {genre} 장르 이야기(한국어)를 작성하세요.

[독자 정보]
- 이름: {name}
- 나이: {age}세
- 읽기 난이도: {self.reading_level}

[장르 가이드]
- {guide}

[스토리 구성]
1) 자연스러운 시작
2) 작은 어려움 → 시도/도움 → 해결
3) 장면 전환은 2~3회
4) 대사 2~4회
5) 쉬운 단어 사용

[길이]
- 문장 수: {self.length_hint}

[안전/윤리]
- {self.safety}

[출력 형식]
이야기 본문

제목: (책 제목)
"""


    @torch.inference_mode()
    def generate_story(self, name: str, age: int, genre: str) -> Dict[str, object]:
        assert self.model is not None and self.tokenizer is not None, "먼저 load() 실행 필요!"
        prompt = self._build_prompt(name, age, genre)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            repetition_penalty=self.repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()

        title, full_content = self._split_title_content(text)
        content = self._split_into_chunks(full_content)

        return {
            "title": title,
            "content": content,
        }