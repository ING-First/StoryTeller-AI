# 📖  **StoryTeller**

### 아이맞춤형 AI 동화 생성 및 리딩 서비스

**StoryTeller**는 LLM과 생성형 AI를 활용하여 **아이 맞춤형 동화 생성·삽화·리딩 서비스**를 제공하는  플랫폼입니다.

 입력한 이름, 나이, 장르를 기반으로 동화를 자동으로 생성하고, LoRA 기반 Stable Diffusion으로 삽화를 만들며, ElevenLabs API를 통해 입력한 목소리로 동화를 읽어줍니다.

---

# 프로젝트 메인사진

### 📍 주요 기능

- **동화 생성**: QLoRA Fine-Tuned LLM으로 맞춤형 동화 생성
- **스토리 평가**: LoRA Fine-Tuned LLM을 통해 품질 자동 평가
- **스토리 요약**: 페이지 단위 요약으로 삽화 생성 프롬프트 최적화
- **삽화 생성**: Stable Diffusion + LoRA로 동화 분위기에 맞는 이미지 생성
- **TTS 리딩**: ElevenLabs API로 부모 목소리 기반 오디오 스트리밍
- **DB 관리**: MySQL에 동화/이미지/오디오/로그 저장 및 이어듣기 지원


---

# **시스템 구성 및 아키텍처**
<img width="500" height="657" alt="스크린샷 2025-08-20 19 59 32" src="https://github.com/user-attachments/assets/f7f20d4b-339f-43bd-abfb-43660eb08c19" />



---

# **🛠️ 기술 스택**

| **Frontend**       |  **Backend**     | **AI**     | **Database**     | **TTS**     | **server**     |
|------------|--------------|--------------|--------------|--------------|--------------|
| [![My Skills](https://skillicons.dev/icons?i=react)](https://skillicons.dev) | [![My Skills](https://skillicons.dev/icons?i=fastapi)](https://skillicons.dev) | [![My Skills](https://skillicons.dev/icons?i=pytorch)](https://skillicons.dev) <img width="50" height="50" alt="스크린샷 2025-08-20 20 08 26" src="https://github.com/user-attachments/assets/e660b773-7cf3-484c-878b-afa4eb04356d" /> | [![My Skills](https://skillicons.dev/icons?i=mysql)](https://skillicons.dev) | <img width="70" height="50" alt="스크린샷 2025-08-20 20 11 41" src="https://github.com/user-attachments/assets/5e12cede-e606-40cb-84f5-60509bc6517f" />  | <img width="70" height="50" alt="스크린샷 2025-08-20 20 13 39" src="https://github.com/user-attachments/assets/a556b245-af0b-48eb-ae35-5d2feaeedc4e" /> | 
