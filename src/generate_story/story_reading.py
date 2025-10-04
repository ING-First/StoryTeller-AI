from __future__ import annotations
from typing import List, Optional, Dict, Any, Union
from datetime import date
import json, os, re
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from db.db_models import FairyTale, FairyTaleLog
from generate_story.generate_sound import SoundGenerator

def _as_pages(contents: Union[List[str], str, bytes, None]) -> List[str]:
    print(f"[DEBUG] _as_pages 호출됨. contents 타입: {type(contents)}, 값: {contents}")
    
    # 페이지 단위로 분리
    if contents is None:
        print("[DEBUG] contents가 None임")
        return []
    if isinstance(contents, bytes):
        try:
            contents = contents.decode("utf-8")
        except Exception:
            print("[DEBUG] bytes → str 변환 실패")
            return []

    if isinstance(contents, list):
        result = [p.strip() for p in contents if (p or "").strip()]
        print(f"[DEBUG] list 형태로 처리됨. 결과: {result}")
        return result
    
    # TEXT에 JSON 배열 문자열로 저장된 경우
    text = str(contents).strip()
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
    
    for i, page in enumerate(pages[:3]):  # 처음 3개 페이지만 로그
        print(f"[DEBUG] 페이지 {i+1}: {page[:100]}...")
    
    return pages


class StoryReader:
    def __init__(self):
        self.sg = SoundGenerator()
        print("[DEBUG] StoryReader 초기화됨")

    def resume_reading(self, db: Session, uid: int, fid: int) -> Dict[str, Any]:
        print(f"[DEBUG] resume_reading 호출됨. uid: {uid}, fid: {fid}")
        ft = self._get_fairy_tale_or_404(db, uid, fid)
        pages = _as_pages(ft.contents)
        total_pages = len(pages)
        total_clips = (total_pages + 1) // 2  # 전체 Clip 수 계산
        print(f"[DEBUG] 총 페이지 수: {total_pages}, 총 Clip 수: {total_clips}")

        log = (
            db.query(FairyTaleLog)
            .filter(FairyTaleLog.uid == uid, FairyTaleLog.fid == fid)
            .order_by(FairyTaleLog.updateDate.desc(), FairyTaleLog.lid.desc())
            .first()
        )
        last_clip = int(getattr(log, "clip", 0) or 0) if log else 0
        # 마지막으로 읽은 Clip을 반환
        resume_clip = last_clip if last_clip > 0 else 1
        print(f"[DEBUG] 마지막 Clip: {last_clip}, 이어읽기 Clip: {resume_clip}")

        return {
            "uid": uid,
            "fid": fid,
            "total_pages": total_pages,
            "last_page": last_clip,
            "next_page": resume_clip,  # 마지막으로 읽은 Clip 반환
        }

    # 특정 페이지 읽기
    def stream_page(
        self,
        db: Session,
        uid: int,
        fid: int,
        page: int,
        ref_wav: Optional[str] = "ref_audio.wav",
    ) -> StreamingResponse:
        ft = self._get_fairy_tale_or_404(db, uid, fid)
        pages = _as_pages(ft.contents)

        if not pages:
            raise HTTPException(status_code=400, detail="동화에 페이지가 없습니다.")

        if not isinstance(page, int) or page < 1 or page > len(pages):
            raise HTTPException(status_code=400, detail=f"잘못된 페이지 번호: 1~{len(pages)}")

        text = (pages[page - 1] or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="선택한 페이지 내용이 비어있습니다.")

        # clip 업데이트
        print(f"[DEBUG] 읽기 로그 업데이트 시작")
        log = (
            db.query(FairyTaleLog)
            .filter(FairyTaleLog.uid == uid, FairyTaleLog.fid == fid)
            .order_by(FairyTaleLog.updateDate.desc(), FairyTaleLog.lid.desc())
            .first()
        )
        try:
            # page를 clip으로 변환 (페이지 1-2 = Clip 1, 페이지 3-4 = Clip 2)
            clip_number = (page + 1) // 2
            
            if not log:
                print(f"[DEBUG] 새 로그 생성. page: {page}, clip: {clip_number}")
                log = FairyTaleLog(
                    uid=uid, fid=fid, clip=clip_number,
                    createDate=date.today(), updateDate=date.today()
                )
                db.add(log); db.flush(); db.refresh(log)
            else:
                old_clip = int(getattr(log, "clip", 0) or 0)
                new_clip = max(old_clip, clip_number)
                print(f"[DEBUG] 로그 업데이트. page: {page}, 이전 clip: {old_clip}, 새 clip: {new_clip}")
                log.clip = new_clip
                log.updateDate = date.today()
            db.commit()
            print("[DEBUG] 로그 업데이트 완료")
        except Exception as e:
            print(f"[DEBUG] 로그 업데이트 실패: {e}")
            db.rollback()
            raise HTTPException(status_code=500, detail=f"log_update_failed: {e}")

        # Zonos TTS 실시간 스트리밍
        if not ref_wav or not os.path.isfile(ref_wav):  
            raise HTTPException(status_code=400, detail=f"참조 오디오 파일 없음: {ref_wav}")
        print(f"[DEBUG] Zonos TTS 스트리밍 시작. ref_wav={ref_wav}, 텍스트 길이={len(text)}")
        return StreamingResponse(
            self.sg.tts_generator(ref_wav=ref_wav, text=text),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'inline; filename="fid{fid}_page{page}.wav"',
                "X-Total-Pages": str(len(pages)),
                "X-Current-Page": str(page),
            },
        )


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