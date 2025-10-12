import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import tempfile
import os
import base64
from db.db_models import Voices

class SoundGenerator:
    def __init__(self, device: str = "cuda"):
        print("[DEBUG] Zonos 모델 로드 중...")
        self.device = device
        self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
        print("[DEBUG] Zonos 모델 로드 완료"  )

        cwd = os.getcwd()
        print(f"[DEBUG] 현재 작업 디렉토리: {cwd}")

        ref_dir = os.path.join(cwd, "ref_voices")
        print(f"[DEBUG] ref_voices 경로 존재 여부: {os.path.exists(ref_dir)} ({ref_dir})")

    # TTS 오디오 스트리밍
    def tts_generator(self, db, text: str, voice_id: str):
        print(f"[DEBUG] tts_generator 호출됨. voice_id={voice_id}, text={text[:50]}...")

        print(f"[DEBUG] 현재 함수 실행 경로: {os.getcwd()}")

        # 음성파일 불러오기
        voice_record = db.query(Voices).filter(Voices.voice_id == voice_id).first()
        print(f"[DEBUG] DB 조회 결과: {voice_record.voiceFile if voice_record else '없음'}")
        if not voice_record:
          raise FileNotFoundError(f"[ERROR] DB에 voice_id={voice_id} 해당 음성이 없습니다.")

        ref_wav_path = voice_record.voiceFile

        if not os.path.isabs(ref_wav_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            ref_wav_path = os.path.normpath(os.path.join(base_dir, "..", ref_wav_path))

        if not os.path.exists(ref_wav_path):
            raise FileNotFoundError(f"[ERROR] ref_wav not found: {ref_wav_path}")

        print(f"[DEBUG] 실제 오디오 경로: {ref_wav_path}")

        wav, sampling_rate = torchaudio.load(ref_wav_path)
        speaker = self.model.make_speaker_embedding(wav, sampling_rate)

        # 조건 설정 (한국어)
        cond_dict = make_cond_dict(
            text=text,
            speaker=speaker,
            language="ko"
        )
        conditioning = self.model.prepare_conditioning(cond_dict)

        # 코드 생성 + 오디오 복호화
        codes = self.model.generate(conditioning)