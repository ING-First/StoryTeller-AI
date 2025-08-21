from dotenv import load_dotenv
import os
import requests
from typing import Generator, Optional

class SoundGenerator:
    def __init__(self):
        load_dotenv(override=False)
        
        self.API_KEY = os.getenv("API_KEY")
        if not self.API_KEY:
            raise RuntimeError("API_KEY가 설정되지 않았습니다.")
        self.ADD_URL = "https://api.elevenlabs.io/v1/voices/add"
        self.TTS_BASE = "https://api.elevenlabs.io/v1/text-to-speech"
        
    def register(self, audio_path, uid, url: Optional[str]=None) -> dict:
        url = url or self.ADD_URL
        data = {
            "name": f"{uid} voice",
            "description": "동화책 TTS 커스텀 목소리",
        }
        headers = {
            "xi-api-key": self.API_KEY
        }
        
        with open(audio_path, "rb") as f:
            files = {"files": (os.path.basename(audio_path), f, "audio/mpeg")}
            response = requests.post(url, headers=headers, data=data, files=files, timeout=60)


        if response.status_code != 200:
            raise Exception(f"Voice register failed: {response.text}")
        
        voice_id = (response.json() or {}).get("voice_id")
        if not voice_id:
            raise Exception("voice_id가 응답에 없습니다.")
        
        return {
          "uid": uid,
          "voice_id": voice_id
        }
        
    def tts_generator(self, voice_id: str, text: str) -> Generator[bytes, None, None]:
        url = f"{self.TTS_BASE}/{voice_id}/stream"
        headers = {
          "xi-api-key": self.API_KEY,
          "Accept": "audio/mpeg",
          "Content-Type": "application/json"
        }
        
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 1.0, "similarity_boost": 0.8},
        }
        
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=None)
        
        if response.status_code != 200:
            raise Exception(f"TTS 생성이 실패하였습니다.: {response.text}")
        
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                yield chunk