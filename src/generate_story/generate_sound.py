import torch
import torchaudio
import subprocess
import os
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from db.db_models import Voices

class SoundGenerator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        print("[DEBUG] Zonos 모델 로드 중...")
        self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)  # 수정됨
        print("[DEBUG] Zonos 모델 로드 완료")

    def _ensure_wav_pcm16_mono_22050(self, wav, sr):
        """WAV 형식 통일 보조 함수"""
        if wav.ndim > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 22050:
            wav = torchaudio.functional.resample(wav, sr, 22050)
        return wav, 22050

    def tts_generator(self, db, text: str, voice_id: str):
        try:
            print("[DEBUG] TTS 요청 시작")

            # DB 조회
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

            # DB 세션 정리
            db.expunge_all()
            db.close()

            # 파일 포맷 감지
            file_check = subprocess.run(["file", "-b", ref_wav_path], capture_output=True, text=True)
            is_webm = "WebM" in file_check.stdout or "Opus" in file_check.stdout

            if is_webm:
                print("[DEBUG] WebM/Opus 형식 감지 → WAV로 변환(ffmpeg)")
                converted_path = ref_wav_path.replace(".wav", "_converted.wav")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", ref_wav_path, "-ar", "22050", "-ac", "1", converted_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                ref_wav_path = converted_path
                print(f"[DEBUG] 변환 완료: {ref_wav_path}")
            else:
                print("[DEBUG] WAV 파일로 확인됨 — 변환 불필요")

            # 오디오 로드
            wav, sampling_rate = torchaudio.load(ref_wav_path, backend="soundfile")
            wav = wav.to(self.device)
            print(f"[DEBUG] 입력 wav 로드 완료: shape={tuple(wav.shape)}, sr={sampling_rate}")

            # 입력 음성 정규화
            wav, sampling_rate = self._ensure_wav_pcm16_mono_22050(wav, sampling_rate)
            print(f"[DEBUG] 입력 wav 정규화 완료: shape={tuple(wav.shape)}, sr={sampling_rate}")

            # 스피커 임베딩
            speaker = self.model.make_speaker_embedding(wav, sampling_rate)
            print("[DEBUG] speaker embedding 생성 완료")

            # conditioning 준비
            cond_dict = make_cond_dict(text=text, speaker=speaker, language="ko")
            conditioning = self.model.prepare_conditioning(cond_dict)
            print("[DEBUG] conditioning 생성 완료")

            # conditioning을 GPU로 이동
            if isinstance(conditioning, dict):
                conditioning = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in conditioning.items()}

            # 🔹 오디오 코드 생성
            print("[DEBUG] 오디오 코드 생성 시작") 
            codes = self.model.generate(conditioning)  
            print("[DEBUG] 오디오 코드 생성 완료") 

            # 🔹 오디오 복원 (디코딩)
            print("[DEBUG] 디코딩 시작")  
            wavs = self.model.autoencoder.decode(codes).cpu() 
            print("[DEBUG] 디코딩 완료") 

            # 출력 저장
            torchaudio.save("/tmp/tts_output.wav", wavs[0], self.model.autoencoder.sampling_rate)
            return open("/tmp/tts_output.wav", "rb")

        except Exception as e:
            print(f"[ERROR] TTS 전체 과정 중 예외 발생: {e}")
            import traceback
            traceback.print_exc()
            raise