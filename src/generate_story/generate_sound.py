import soundfile as sf
import torch
import torchaudio
from torchaudio.functional import resample
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import tempfile
import os
from db.db_models import Voices
import subprocess
import traceback
import shutil
import datetime
from typing import Tuple

class SoundGenerator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        print("[DEBUG] Zonos 모델 로드 중...")
        self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
        print("[DEBUG] Zonos 모델 로드 완료")

        cwd = os.getcwd()
        ref_dir = os.path.join(cwd, "ref_voices")

    def _ensure_wav_pcm16_mono_22050(self, wav_tensor: torch.Tensor, sr: int) -> Tuple[torch.Tensor, int]:
        """
        입력: [C, N] 또는 [N] 텐서, sr
        출력: 모노 [1, N] 텐서, 22050Hz
        """
        # [N] -> [1, N]
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)

        # 다채널 -> 모노
        if wav_tensor.size(0) > 1:
            print(f"[DEBUG] 다채널 입력 감지: {tuple(wav_tensor.shape)} -> 평균으로 모노 변환")
            wav_tensor = wav_tensor.mean(dim=0, keepdim=True)

        # 리샘플
        if sr != 22050:
            print(f"[DEBUG] 샘플레이트 변환: {sr} -> 22050")
            wav_tensor = resample(wav_tensor, orig_freq=sr, new_freq=22050)
            sr = 22050

        # float32로 변환 및 클리핑/정규화
        wav_tensor = wav_tensor.to(torch.float32)
        max_val = torch.max(torch.abs(wav_tensor))
        if torch.isfinite(max_val) and max_val > 1.0:
            print(f"[DEBUG] 정규화 수행 (max={max_val.item():.4f})")
            wav_tensor = wav_tensor / max_val

        # 보수적 클리핑
        wav_tensor = torch.clamp(wav_tensor, -1.0, 1.0)

        # 최종 형태 [1, N]
        if wav_tensor.dim() != 2 or wav_tensor.size(0) != 1:
            raise RuntimeError(f"[BUG] 예상치 못한 오디오 텐서 형태: {tuple(wav_tensor.shape)} (기대: [1, N])")

        return wav_tensor, sr

    def _save_wav_pcm16(self, path: str, mono_wav_22050: torch.Tensor, sr: int = 22050) -> None:
        """
        torchaudio.save로 무조건 PCM_16 모노로 저장. 입력은 [1, N], float32 [-1,1].
        """
        # encoding/bit 지정 - PCM 16bit로 고정
        torchaudio.save(
            path,
            mono_wav_22050.cpu(),
            sample_rate=sr,
            encoding="PCM_S",
            bits_per_sample=16
        )

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

            # 입력 음성 정규화 (모노/22050 보장)
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
                conditioning = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in conditioning.items()}
            else:
                conditioning = conditioning.to(self.device)
            print("[DEBUG] conditioning 디바이스 이동 완료")

            # 오디오 생성
            print("[DEBUG] 오디오 코드 생성 시작")
            with torch.no_grad():
                audio = self.model.generate(conditioning)
            print("[DEBUG] 오디오 코드 생성 완료")

            # 모델 출력 정규화: [1, N], 22050Hz, float32 [-1,1]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            elif audio.dim() == 3:
                # 일부 모델이 [B, C, N] 반환할 수 있어 첫 배치 제거
                audio = audio.squeeze(0)
            audio = audio.to(torch.float32)

            # [C, N] 예상. 채널 처리/리샘플
            if audio.dim() == 2 and audio.size(0) > 1:
                print(f"[DEBUG] 생성 오디오 다채널 감지: {tuple(audio.shape)} -> 모노 변환")
                audio = audio.mean(dim=0, keepdim=True)
            elif audio.dim() == 1:
                audio = audio.unsqueeze(0)

            # 샘플레이트를 모델이 반환하지 않으므로 22050 가정(모델 설정과 일치)
            gen_sr = 22050

            max_val = torch.max(torch.abs(audio))
            if torch.isfinite(max_val) and max_val > 1.0:
                print(f"[DEBUG] 생성 오디오 정규화 수행 (max={max_val.item():.4f})")
                audio = audio / max_val
            audio = torch.clamp(audio, -1.0, 1.0)

            # 출력 경로
            suffix = "webM" if is_webm else "wav"
            output_filename = f"tts_output_{suffix}.wav"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)

            # 저장: 반드시 PCM 16bit 모노
            self._save_wav_pcm16(output_path, audio, sr=gen_sr)
            print(f"[DEBUG] 생성된 오디오 저장 완료: {output_path}")

            # 검증 로그
            try:
                fi = torchaudio.info(output_path)
                print(f"[DEBUG] 저장 검증 - channels={fi.num_channels}, sample_rate={fi.sample_rate}, bits_per_sample={fi.bits_per_sample}")
                file_probe = subprocess.run(["file", "-b", output_path], capture_output=True, text=True)
                print(f"[DEBUG] file 명령 결과: {file_probe.stdout.strip()}")
            except Exception as _e:
                print(f"[WARN] 저장 검증 중 경고: {_e}")

            # 스트리밍 리턴
            def audio_stream():
                with open(output_path, "rb") as f:
                    while True:
                        chunk = f.read(4096)
                        if not chunk:
                            break
                        yield chunk

            return audio_stream()

        except Exception as e:
            print("[ERROR] TTS 전체 과정 중 예외 발생:", e)
            traceback.print_exc()
            raise