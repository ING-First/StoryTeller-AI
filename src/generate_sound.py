from dotenv import load_dotenv
import requests
import os

class SoundGenerator:
    def __init__(self):
        load_dotenv(override=False)
        
        self.API_KEY = os.getenv("API_KEY")
        if not self.API_KEY:
            raise RuntimeError("API_KEY가 설정되지 않았습니다.")
        
    def register(self, audio_path, uid, url="https://api.elevenlabs.io/v1/voices/add"):
        url = url
        files = {
            "files": open(audio_path, "rb"),
        }
        data = {
            "name": f"{uid} voice",
            "description": "동화책 TTS 커스텀 목소리",
        }
        headers = {
            "xi-api-key": self.API_KEY
        }
        
        response = requests.post(url, headers=headers, data=data, files=files)
        if response.status_code != 200:
            raise Exception(f"Voice register failed: {response.text}")
        
        res_json = response.json()
        voice_id = res_json.get("voice_id")
        
        return {
          "uid": uid,
          "voice_id": voice_id
        }
        
    def tts_generator(self, fid, uid, voice_id, contents):
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
          "xi-api-key": self.API_KEY,
          "Accept": "audio/mpeg",
          "Content-Type": "application/json"
        }
        
        data = {
            "text": contents,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code != 200:
            raise Exception(f"TTS 생성이 실패하였습니다.: {response.text}")
        
        output_path = f"output_{fid}.mp3"
        with open(output_path, "wb") as f:
            f.write(response.content)

        return {
          "fid": fid,
          "uid": uid,
          "contents": contents,
          "output_path": output_path          
        }