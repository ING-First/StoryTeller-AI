import os
import requests
from typing import Optional, Type
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

def create_generate_router(
    GenerateStoryRequest: Type[BaseModel],
    GenerateStoryResponse: Type[BaseModel],
    remote_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> APIRouter:

    REMOTE_GENERATE_URL = remote_url or os.getenv("REMOTE_GENERATE_URL")
    if REMOTE_GENERATE_URL is None or not REMOTE_GENERATE_URL.strip():
        raise RuntimeError(
            "REMOTE_GENERATE_URL이 설정되지 않았습니다. "
            "예) export REMOTE_GENERATE_URL='https://<colab-or-ngrok>/generate'"
        )
    try:
        REQUEST_TIMEOUT = float(timeout if timeout is not None else os.getenv("REMOTE_TIMEOUT", "5"))
    except ValueError:
        REQUEST_TIMEOUT = 5.0

    router = APIRouter(tags=["generate-proxy"])

    # 내부 호출 유틸 (원격 서버로 포워딩)
    def _call_remote_generate(payload: BaseModel):
        try:
            resp = requests.post(
                REMOTE_GENERATE_URL,
                json=payload.dict(),
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as e:
            # 네트워크/타임아웃 등
            raise HTTPException(status_code=502, detail=f"upstream_unreachable: {e}")

        if resp.status_code != 200:
            # 원격 서버에서 에러 응답 시 본문 일부 포함
            text = resp.text
            if len(text) > 300:
                text = text[:300] + "..."
            raise HTTPException(status_code=502, detail=f"upstream_http_{resp.status_code}: {text}")

        try:
            data = resp.json()
        except ValueError:
            raise HTTPException(status_code=502, detail="upstream_invalid_json")

        # 원격 응답을 검증된 스키마로 반환
        try:
            return GenerateStoryResponse(**data)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"upstream_schema_mismatch: {e}")

    # 프록시 엔드포인트
    @router.post("/generate", response_model=GenerateStoryResponse)
    def generate_proxy(req: GenerateStoryRequest):  # type: ignore[valid-type]
        """
        프론트 → (로컬) /generate → (프록시) → 원격 코랩 /generate
        """
        return _call_remote_generate(req)

    return router