import soundfile as sf
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import tempfile
import os
import base64
from db.db_models import Voices
import subprocess
import traceback

class SoundGenerator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

        cwd = os.getcwd()
        ref_dir = os.path.join(cwd, "ref_voices")

    def tts_generator(self, db, text: str, voice_id: str):

        try:
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

            # DB 세션 조기 종료 
            db.expunge_all()
            db.close()

            # 파일 포맷 감지
            file_check = subprocess.run(["file", "-b", ref_wav_path], capture_output=True, text=True)
            if "WebM" in file_check.stdout or "Opus" in file_check.stdout:
                print("[DEBUG] WebM 형식 감지됨 → WAV로 변환 시작")
                converted_path = ref_wav_path.replace(".wav", "_converted.wav")
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", ref_wav_path,
                    "-ar", "22050", "-ac", "1", 
                    converted_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                ref_wav_path = converted_path
                print(f"[DEBUG] 변환 완료: {ref_wav_path}")
            else:
                print("[DEBUG] WAV 파일로 확인됨 — 변환 불필요")

            # 오디오 로드 및 디바이스 이동
            wav, sampling_rate = torchaudio.load(ref_wav_path, backend="soundfile")
            wav = wav.to(self.device)  
            print("[DEBUG] wav 로드 및 디바이스 이동 완료")

            # 스피커 임베딩 생성
            speaker = self.model.make_speaker_embedding(wav, sampling_rate)
            print("[DEBUG] speaker embedding 생성 완료")

            # conditioning 준비
            cond_dict = make_cond_dict(
                text=text,
                speaker=speaker,
                language="ko"
            )
            conditioning = self.model.prepare_conditioning(cond_dict)
            print("[DEBUG] conditioning 생성 완료")

            # conditioning을 GPU로 이동  
            if isinstance(conditioning, dict):
                conditioning = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in conditioning.items()
                }
            else:
                conditioning = conditioning.to(self.device)
            print("[DEBUG] conditioning 디바이스 이동 완료")

            # 오디오 생성 전체 try
            print("[DEBUG] 오디오 코드 생성 시작")
            audio = self.model.generate(conditioning)
            print("[DEBUG] 오디오 코드 생성 완료")

            # CPU로 이동 후 저장
            output_path = os.path.join(tempfile.gettempdir(), "tts_output.wav")
            audio = audio.cpu() 

            if audio.dim() == 1:
                audio = audio.unsqueeze(0) 
            elif audio.dim() == 3:
                audio = audio.squeeze(0)
            if audio.dtype != torch.float32:
                audio = audio.to(torch.float32)

            sf.write(output_path, audio.squeeze(0).numpy(), 22050)
            print(f"[DEBUG] 생성된 오디오 저장 완료: {output_path}")      

            def audio_stream():
                with open(output_path, "rb") as f:
                    while chunk := f.read(4096):
                        yield chunk

            return audio_stream()


        except Exception as e:
            print("[ERROR] TTS 전체 과정 중 예외 발생:", e)
            traceback.print_exc()
            raise
