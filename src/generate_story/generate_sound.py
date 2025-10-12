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
        print("[DEBUG] Zonos 모델 로드 완료")

    # TTS 오디오 스트리밍
    def tts_generator(self, db, text: str, voice_id: str):
        print(f"[DEBUG] tts_generator 호출됨. voice_id={voice_id}, text={text[:50]}...")

        # 음성파일 불러오기
        voice_record = db.query(Voices).filter(Voices.voice_id == voice_id).first()
        if not voice_record:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            ref_wav_path = os.path.join(base_dir, "..", ref_wav_path)
            ref_wav_path = os.path.normpath(ref_wav_path)
            print(f"[DEBUG] 절대경로 보정 완료 → {ref_wav_path}")

        ref_wav_path = voice_record.voiceFile

        if not os.path.exists(ref_wav_path):
            raise FileNotFoundError(f"[ERROR] ref_wav not found: {ref_wav_path}")

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
        wavs = self.model.autoencoder.decode(codes).cpu()

        # [1, time] 형태 보장
        audio_tensor = wavs[0].view(1, -1)

        # 임시 wav 파일 저장 
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        torchaudio.save(tmpfile.name, audio_tensor, self.model.autoencoder.sampling_rate)

        print(f"[DEBUG] TTS 생성 완료: {tmpfile.name}, shape={audio_tensor.shape}")

        # 파일 스트리밍
        def iterfile():
            with open(tmpfile.name, "rb") as f:
                yield from f
            os.remove(tmpfile.name)

        return iterfile()