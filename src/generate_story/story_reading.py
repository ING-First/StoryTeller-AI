from __future__ import annotations
from typing import List, Optional, Dict, Any, Union
from datetime import date
import json
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
            print(f"[DEBUG] JSON 배열로 파싱됨. 결과: {result}")
            return result
    except Exception as e:
        print(f"[DEBUG] JSON 파싱 실패: {e}")
        print("[DEBUG] 문자열을 문장 단위로 분할 시도")
        
    # 문자열을 문장 단위로 분할 후 2문장씩 묶기
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 2문장씩 묶어서 페이지 생성
    pages = []
    for i in range(0, len(sentences), 2):
        page_sentences = sentences[i:i+2]  # 2개씩 가져오기
        page_text = ' '.join(page_sentences)
        pages.append(page_text)
    
    print(f"[DEBUG] 2문장씩 묶은 결과: {len(pages)}개 페이지")
    for i, page in enumerate(pages[:3]):  # 처음 3개 페이지만 로그
        print(f"[DEBUG] 페이지 {i+1}: {page[:100]}...")
    
    return pages


class StoryReader:

    def __init__(self):
        self.sg = SoundGenerator()
        print("[DEBUG] StoryReader 초기화됨")

    # 이어듣기 상태 조회
    def resume_reading(self, db: Session, uid: int, fid: int) -> Dict[str, Any]:
        print(f"[DEBUG] resume_reading 호출됨. uid: {uid}, fid: {fid}")
        ft = self._get_fairy_tale_or_404(db, uid, fid)
        pages = _as_pages(ft.contents)
        total_pages = len(pages)
        print(f"[DEBUG] 총 페이지 수: {total_pages}")

        log = (
            db.query(FairyTaleLog)
            .filter(FairyTaleLog.uid == uid, FairyTaleLog.fid == fid)
            .order_by(FairyTaleLog.updateDate.desc(), FairyTaleLog.lid.desc())
            .first()
        )
        last_page = int(getattr(log, "clip", 0) or 0) if log else 0  # clip=책갈피
        next_page = min(last_page + 1, total_pages) if total_pages > 0 else 0
        print(f"[DEBUG] 마지막 페이지: {last_page}, 다음 페이지: {next_page}")

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
        print(f"[DEBUG] stream_page 호출됨. uid: {uid}, fid: {fid}, page: {page}, voice_id: {voice_id}")
        
        ft = self._get_fairy_tale_or_404(db, uid, fid)
        print(f"[DEBUG] 동화 찾음. title: {ft.title}")
        
        pages = _as_pages(ft.contents)
        print(f"[DEBUG] 페이지 변환 완료. 총 {len(pages)}개 페이지")
        
        if not pages:
            print("[DEBUG] 페이지가 비어있음")
            raise HTTPException(status_code=400, detail="동화에 페이지가 없습니다. (pages 비어있음)")

        if page < 1 or page > len(pages):
            print(f"[DEBUG] 잘못된 페이지 번호. 요청: {page}, 범위: 1~{len(pages)}")
            raise HTTPException(status_code=400, detail=f"잘못된 페이지 번호: 1~{len(pages)}")

        text = (pages[page - 1] or "").strip()
        print(f"[DEBUG] 페이지 텍스트 (길이: {len(text)}): {text[:100]}...")
        
        if not text:
            print("[DEBUG] 페이지 텍스트가 비어있음")
            raise HTTPException(status_code=400, detail="선택한 페이지 내용이 비어있습니다.")

        print(f"[DEBUG] voice_id 처리 전: {voice_id}")
        vid = voice_id or self._get_user_default_voice_id(db, uid)
        print(f"[DEBUG] 최종 voice_id: {vid}")
        
        if not vid:
            print("[DEBUG] voice_id가 None임")
            raise HTTPException(status_code=400, detail="voice_id 없음. 먼저 /voices/register를 호출하세요.")

        # clip 업데이트
        print(f"[DEBUG] 읽기 로그 업데이트 시작")
        log = (
            db.query(FairyTaleLog)
            .filter(FairyTaleLog.uid == uid, FairyTaleLog.fid == fid)
            .order_by(FairyTaleLog.updateDate.desc(), FairyTaleLog.lid.desc())
            .first()
        )
        try:
            if not log:
                print("[DEBUG] 새 로그 생성")
                log = FairyTaleLog(
                    uid=uid, fid=fid, clip=page,
                    createDate=date.today(), updateDate=date.today()
                )
                db.add(log); db.flush(); db.refresh(log)
            else:
                old_clip = int(getattr(log, "clip", 0) or 0)
                new_clip = max(old_clip, page)
                print(f"[DEBUG] 로그 업데이트. 이전 clip: {old_clip}, 새 clip: {new_clip}")
                log.clip = new_clip
                log.updateDate = date.today()
            db.commit()
            print("[DEBUG] 로그 업데이트 완료")
        except Exception as e:
            print(f"[DEBUG] 로그 업데이트 실패: {e}")
            db.rollback()
            raise HTTPException(status_code=500, detail=f"log_update_failed: {e}")

        # ElevenLabs TTS 실시간 스트리밍
        print(f"[DEBUG] TTS 스트리밍 시작. voice_id: {vid}, 텍스트 길이: {len(text)}")
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
        print(f"[DEBUG] _get_user_default_voice_id 호출됨. uid: {uid}")
        v = (
            db.query(Voices)
            .filter(Voices.uid == uid)
            .order_by(Voices.createDate.desc())
            .first()
        )
        result = getattr(v, "voice_id", None) if v else None
        print(f"[DEBUG] 사용자 기본 voice_id: {result}")
        return result

    def _get_fairy_tale_or_404(self, db: Session, uid: int, fid: int) -> FairyTale:
        print(f"[DEBUG] _get_fairy_tale_or_404 호출됨. uid: {uid}, fid: {fid}")
        ft = (
            db.query(FairyTale)
            .filter(FairyTale.fid == fid)
            .first()
        )
        if not ft:
            print("[DEBUG] 동화를 찾을 수 없음")
            raise HTTPException(status_code=404, detail="해당 동화를 찾을 수 없습니다.")
        print(f"[DEBUG] 동화 찾음. title: {ft.title}, contents 타입: {type(ft.contents)}")
        return ft