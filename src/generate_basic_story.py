from sqlalchemy.orm import Session
from db.db_connector import SessionLocal
from db.db_models import FairyTale, FairyTaleImage
from generate_story.generate_image import ImageGenerator
from generate_story.generate_summary import Summarizer
from tqdm import tqdm
from datetime import date

class BasicStoryGenerator:
    def __init__(self):
        self.summarizer = None
        self.img_generator = None

        try:
            self.db = SessionLocal()
        except Exception as e:
            self.db.close()

    def load_model(self):
        # 요약 모델 로드
        self.summarizer = Summarizer()
        self.summarizer.load_lora_model()

        # 스테이블 디퓨전 모델 로드
        self.img_generator = ImageGenerator()
        self.img_generator.load_diffusion_model()

    
    def search_story(self):
        return self.db.query(FairyTale).filter(FairyTale.uid == 0).all()
    
    def main(self):
        self.load_model()

        stories = self.search_story()
        for story in stories:
            page_summaries = self.summarizer.generate_page_summaries(story.contents)
            for s in tqdm(page_summaries):
                image_path, file_name = self.img_generator.generate_image(s, story.title)

                images = FairyTaleImage(
                    fid=story.fid,
                    image_path=image_path,
                    file_name=file_name,
                    createDate=date.today(),
                )
                
                try:
                    self.db.add(images)
                    self.db.commit()
                    self.db.refresh(images)
                except Exception as e:
                    self.db.rollback()
                    print("이미지 데이터 저장 실패")
    

if __name__ == "__main__":
    basic_story_generator = BasicStoryGenerator()
    basic_story_generator.main()
            