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
        self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
        print("[DEBUG] Zonos 모델 로드 완료")

    def _ensure_wav_pcm16_mono_22050(self, wav, sr):
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
            if not voice_record:
                raise FileNotFoundError(f"[ERROR] DB에 voice_id={voice_id} 해당 음성이 없습니다.")

            print(f"[DEBUG] DB 조회 결과: {voice_record.voiceFile}")
            base_dir = os.path.dirname(os.path.abspath(__file__))
            ref_wav_path = os.path.normpath(os.path.join(base_dir, "..", voice_record.voiceFile))

            if not os.path.exists(ref_wav_path):
                raise FileNotFoundError(f"[ERROR] 원본 파일 없음: {ref_wav_path}")

            # 🔹 DB 경로 기반으로 ref_audio 저장 경로 자동 생성
            ref_audio_relpath = voice_record.voiceFile.replace("ref_voices", "ref_audio").replace(".webm", "_converted.wav")
            ref_audio_path = os.path.normpath(os.path.join(base_dir, "..", ref_audio_relpath))
            os.makedirs(os.path.dirname(ref_audio_path), exist_ok=True)

            print(f"[DEBUG] 변환 후 저장 경로: {ref_audio_path}")

            # 파일 포맷 감지
            file_check = subprocess.run(["file", "-b", ref_wav_path], capture_output=True, text=True)
            is_webm = "WebM" in file_check.stdout or "Opus" in file_check.stdout

            if is_webm:
                print("[DEBUG] WebM/Opus 형식 감지 → WAV로 변환(ffmpeg)")
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", ref_wav_path,
                    "-ar", "22050",
                    "-ac", "1",
                    "-acodec", "pcm_s16le",
                    ref_audio_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                ref_wav_path = ref_audio_path
                print(f"[DEBUG] 변환 완료: {ref_audio_path}")
            else:
                print("[DEBUG] WAV 파일로 확인됨 — 변환 불필요")
                if ref_wav_path != ref_audio_path:
                    subprocess.run(["cp", ref_wav_path, ref_audio_path])
                    print(f"[DEBUG] WAV 복사 완료: {ref_audio_path}")
                ref_wav_path = ref_audio_path

            # 오디오 로드
            wav, sampling_rate = torchaudio.load(ref_wav_path, backend="soundfile")
            wav = wav.to(self.device)
            print(f"[DEBUG] 입력 wav 로드 완료: shape={tuple(wav.shape)}, sr={sampling_rate}")

            wav, sampling_rate = self._ensure_wav_pcm16_mono_22050(wav, sampling_rate)
            print(f"[DEBUG] 입력 wav 정규화 완료: shape={tuple(wav.shape)}, sr={sampling_rate}")

            # 스피커 임베딩
            speaker = self.model.make_speaker_embedding(wav, sampling_rate)
            print("[DEBUG] speaker embedding 생성 완료")

            # conditioning 준비
            cond_dict = make_cond_dict(text=text, speaker=speaker, language="ko")
            conditioning = self.model.prepare_conditioning(cond_dict)
            print("[DEBUG] conditioning 생성 완료")

            if isinstance(conditioning, dict):
                conditioning = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in conditioning.items()}

            print("[DEBUG] 오디오 코드 생성 시작") 
            codes = self.model.generate(conditioning)  
            print("[DEBUG] 오디오 코드 생성 완료") 

            print("[DEBUG] 디코딩 시작")  
            wavs = self.model.autoencoder.decode(codes).cpu() 
            print("[DEBUG] 디코딩 완료") 

            # 출력 저장
            tts_output_path = "/tmp/tts_output.wav"
            torchaudio.save(tts_output_path, wavs[0], self.model.autoencoder.sampling_rate)
            print(f"[DEBUG] TTS 결과 저장 완료: {tts_output_path}")

            return open(tts_output_path, "rb")

        except Exception as e:
            print(f"[ERROR] TTS 전체 과정 중 예외 발생: {e}")
            import traceback
            traceback.print_exc()
            raise