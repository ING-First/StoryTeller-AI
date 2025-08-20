from __future__ import annotations
from typing import List, Optional, Dict, Any, Union
from datetime import date
import json
import re
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from db.db_models import FairyTale, FairyTaleLog, Voices
from generate_story.generate_sound import SoundGenerator

def _as_pages(contents: Union[List[str], str, None]) -> List[str]:
    print(f"[DEBUG] _as_pages 호출됨. contents 타입: {type(contents)}, 값: {contents}")
    if contents is None:
        print("[DEBUG] contents가 None임")
        return []
    if isinstance(contents, list):
        result = [p.strip() for p in contents if (p or "").strip()]
        print(f"[DEBUG] list 형태로 처리됨. 결과: {result}")
        return result
    
    # TEXT에 JSON 배열 문자열로 저장된 경우
    text = contents.strip()
    if not text:
        print("[DEBUG] contents가 빈 문자열임")
        return []
    
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            result = [str(p).strip() for p in parsed if str(p).strip()]
            return result
    except Exception as e:
        print(f"[DEBUG] JSON 파싱 실패: {e}")
        print("[DEBUG] 문자열을 문장 단위로 분할 시도")
        
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 2문장씩 묶어서 페이지 생성
    pages = []
    for i in range(0, len(sentences), 2):
        page_sentences = sentences[i:i+2]  # 2개씩 가져오기
        page_text = ' '.join(page_sentences)
        pages.append(page_text)
    
    return pages

class StoryReader:

    def __init__(self):
        self.sg = SoundGenerator()

    # 이어듣기 상태 조회
    def resume_reading(self, db: Session, uid: int, fid: int) -> Dict[str, Any]:
        ft = self._get_fairy_tale_or_404(db, uid, fid)
        pages = _as_pages(ft.contents)
        total_pages = len(pages)

        log = (
            db.query(FairyTaleLog)
            .filter(FairyTaleLog.uid == uid, FairyTaleLog.fid == fid)
            .order_by(FairyTaleLog.updateDate.desc(), FairyTaleLog.lid.desc())
            .first()
        )
        last_page = int(getattr(log, "clip", 0) or 0) if log else 0  # clip=책갈피
        next_page = min(last_page + 1, total_pages) if total_pages > 0 else 0

        return {
            "uid": uid,
            "fid": fid,
            "total_pages": total_pages,
            "last_page": last_page,
            "next_page": next_page,
        }

    # 특정 페이지 읽기
    def stream_page(
        self,
        db: Session,
        uid: int,
        fid: int,
        page: int,
        voice_id: Optional[str] = None,
    ) -> StreamingResponse:
        ft = self._get_fairy_tale_or_404(db, uid, fid)
        pages = _as_pages(ft.contents)
        if not pages:
            raise HTTPException(status_code=400, detail="동화에 페이지가 없습니다. (pages 비어있음)")

        if page < 1 or page > len(pages):
            raise HTTPException(status_code=400, detail=f"잘못된 페이지 번호: 1~{len(pages)}")

        text = (pages[page - 1] or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="선택한 페이지 내용이 비어있습니다.")

        vid = voice_id or self._get_user_default_voice_id(db, uid)
        if not vid:
            raise HTTPException(status_code=400, detail="voice_id 없음. 먼저 /voices/register를 호출하세요.")

        # clip
        log = (
            db.query(FairyTaleLog)
            .filter(FairyTaleLog.uid == uid, FairyTaleLog.fid == fid)
            .order_by(FairyTaleLog.updateDate.desc(), FairyTaleLog.lid.desc())
            .first()
        )
        try:
            if not log:
                log = FairyTaleLog(
                    uid=uid, fid=fid, clip=page,
                    createDate=date.today(), updateDate=date.today()
                )
                db.add(log); db.flush(); db.refresh(log)
            else:
                log.clip = max(int(getattr(log, "clip", 0) or 0), page)
                log.updateDate = date.today()
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"log_update_failed: {e}")

        # ElevenLabs TTS 실시간 스트리밍
        return StreamingResponse(
            self.sg.tts_generator(voice_id=vid, text=text),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f'inline; filename="fid{fid}_page{page}.mp3"',
                "X-Total-Pages": str(len(pages)),
                "X-Current-Page": str(page),
            },
        )

    def _get_user_default_voice_id(self, db: Session, uid: int) -> Optional[str]:
        v = (
            db.query(Voices)
            .filter(Voices.uid == uid)
            .order_by(Voices.createDate.desc())
            .first()
        )
        return getattr(v, "voice_id", None) if v else None

    def _get_fairy_tale_or_404(self, db: Session, uid: int, fid: int) -> FairyTale:
        print(f"[DEBUG] _get_fairy_tale_or_404 호출됨. uid: {uid}, fid: {fid}")
        ft = (
            db.query(FairyTale)
            .filter(FairyTale.fid == fid)
            .first()
        )
        if not ft:
            raise HTTPException(status_code=404, detail="해당 동화를 찾을 수 없습니다.")
        return ft