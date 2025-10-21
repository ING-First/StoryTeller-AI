from sqlalchemy.orm import Session
from db.db_connector import SessionLocal
from db.db_models import FairyTale, FairyTaleImages
from generate_story.lora_manager import get_lora_manager, ensure_model_loaded
from generate_story.generate_image import ImageGenerator
from generate_story.generate_summary import Summarizer
from tqdm import tqdm
from datetime import date
import time
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU 0 사용

import torch
# GPU 강제 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicStoryGenerator:
    def __init__(self):
        self.summarizer = None
        self.img_generator = None
        self.lora_manager = None
        self.db = None

        try:
            self.db = SessionLocal()
            logger.info("데이터베이스 연결 성공")
        except Exception as e:
            logger.error(f"데이터베이스 연결 실패: {e}")
            if self.db:
                self.db.close()
            raise

    def load_models(self):
        """모든 필요한 모델 로드"""
        logger.info("모델 로딩 시작...")
        
        # LoRA Manager 초기화 및 베이스 모델 로딩 대기
        logger.info("LoRA Manager 초기화 중...")
        self.lora_manager = get_lora_manager()
        
        # 베이스 모델 로딩 대기
        while not ensure_model_loaded():
            time.sleep(1)
            logger.info("베이스 모델 로딩 대기중...")
        
        logger.info("베이스 모델 로딩 완료")

        # 요약 모델 로드
        logger.info("요약 모델 로딩 중...")
        self.summarizer = Summarizer()
        logger.info("요약 모델 로딩 완료")

        # 스테이블 디퓨전 모델 로드
        logger.info("이미지 생성 모델 로딩 중...")
        self.img_generator = ImageGenerator()
        self.img_generator.load_diffusion_model()
        logger.info("이미지 생성 모델 로딩 완료")
        
        logger.info("모든 모델 로딩 완료!")

    def search_stories(self):
        """uid=0인 기본 동화 검색"""
        try:
            stories = self.db.query(FairyTale).filter(FairyTale.uid == 0).all()
            logger.info(f"검색된 기본 동화 수: {len(stories)}")
            return stories
        except Exception as e:
            logger.error(f"동화 검색 실패: {e}")
            return []

    def generate_images_for_story(self, story: FairyTale):
        """특정 동화에 대한 이미지 생성"""
        try:
            logger.info(f"동화 '{story.title}' (fid={story.fid}) 이미지 생성 시작")
            
            # 이미 이미지가 있는지 확인
            existing_images = self.db.query(FairyTaleImages).filter(
                FairyTaleImages.fid == story.fid
            ).count()
            
            if existing_images > 0:
                logger.info(f"동화 {story.fid}는 이미 {existing_images}개의 이미지가 있습니다. 스킵합니다.")
                return
            
            # contents 처리: 2문장씩 분할
            logger.info("본문을 2문장씩 분할 중...")
            
            # 문장 분리
            def split_sentences_kor(s: str):
                import re
                s = re.sub(r"\s+", " ", s.strip())
                pat = re.compile(r'.*?(?:다\.|요\.|[.!?…])')
                sentences = pat.findall(s)
                tail = s[sum(len(x) for x in sentences):].strip()
                if tail: 
                    sentences.append(tail)
                return [x.strip() for x in sentences if x.strip()]
            
            # 2문장씩 묶기
            sentences = split_sentences_kor(story.contents)
            content_chunks = []
            for i in range(0, len(sentences), 2):
                chunk = " ".join(sentences[i:i+2]).strip()
                if chunk:
                    content_chunks.append(chunk)
            
            logger.info(f"총 {len(content_chunks)}개 청크로 분할됨")
            
            # 페이지 요약 생성
            logger.info("페이지 요약 생성 중...")
            page_summaries = self.summarizer.generate_page_summaries(content_chunks)
            
            # 각 페이지에 대한 이미지 생성
            logger.info(f"총 {len(page_summaries)}개 이미지 생성 중...")
            for idx, summary in enumerate(tqdm(page_summaries, desc=f"'{story.title}' 이미지 생성")):
                try:
                    # 이미지 생성
                    image_path, file_name = self.img_generator.generate_image(summary, story.title)
                    
                    # DB에 저장
                    image_record = FairyTaleImages(
                        fid=story.fid,
                        image_path=image_path,
                        file_name=file_name,
                        createDate=date.today(),
                    )
                    
                    self.db.add(image_record)
                    self.db.commit()
                    self.db.refresh(image_record)
                    
                    logger.info(f"이미지 {idx + 1}/{len(page_summaries)} 저장 완료: {file_name}")
                    
                except Exception as e:
                    self.db.rollback()
                    logger.error(f"이미지 {idx + 1} 생성/저장 실패: {e}")
                    continue
            
            logger.info(f"동화 '{story.title}' 이미지 생성 완료!")
            
        except Exception as e:
            logger.error(f"동화 {story.fid} 이미지 생성 중 오류: {e}")
            self.db.rollback()

    def main(self):
        """메인 실행 함수"""
        try:
            logger.info("=" * 50)
            logger.info("기본 동화 이미지 생성 프로세스 시작")
            logger.info("=" * 50)
            
            # 모델 로드
            self.load_models()
            
            # 기본 동화 검색
            stories = self.search_stories()
            
            if not stories:
                logger.warning("생성할 기본 동화가 없습니다.")
                return
            
            # 각 동화에 대해 이미지 생성
            for idx, story in enumerate(stories, 1):
                logger.info(f"\n[{idx}/{len(stories)}] 동화 처리 중...")
                self.generate_images_for_story(story)
            
            logger.info("=" * 50)
            logger.info("모든 기본 동화 이미지 생성 완료!")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"메인 프로세스 실행 중 오류: {e}")
            raise
        finally:
            if self.db:
                self.db.close()
                logger.info("데이터베이스 연결 종료")


if __name__ == "__main__":
    try:
        basic_story_generator = BasicStoryGenerator()
        basic_story_generator.main()
    except KeyboardInterrupt:
        logger.info("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"프로그램 실행 실패: {e}")
        raise