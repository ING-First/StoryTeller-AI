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

---
# 라이선스 안내

이 프로젝트는 **Apache License 2.0**에 따라 배포됩니다.  
Apache 2.0 라이선스의 전문은 [여기에서 확인할 수 있습니다](http://www.apache.org/licenses/LICENSE-2.0).

본 프로젝트에는 다음과 같은 오픈소스 소프트웨어가 포함되어 있으며,  
각 구성요소는 해당 라이선스 조건을 따릅니다:

- MIT License: bitsandbytes, sqlalchemy, jose, httpx, pydantic  
- BSD 3-Clause License: numpy, pandas, passlib, dotenv  
- PSF License: Python 표준 라이브러리(os, re, logging 등)  
- Zope Public License: datetime  

또한 본 프로젝트는 아래의 AI 모델을 포함하고 있습니다:

> - **Bingsu/my-korean-stable-diffusion-v1-5 모델을 파인튜닝 하여 재배포됩니다**  
>   이 모델은 **CreativeML OpenRAIL-M 라이선스**에 따라 배포되며,  
>   사용자에게 윤리적·법적 사용 조건을 부과합니다.  
>   (예: 불법적, 유해한, 차별적 콘텐츠 생성 금지 등)

모델 사용자는 반드시 다음 라이선스 전문 및 사용 정책을 확인해야 합니다:  
👉 [CreativeML OpenRAIL-M 라이선스 전문 보기](https://huggingface.co/spaces/CompVis/stable-diffusion-license)

