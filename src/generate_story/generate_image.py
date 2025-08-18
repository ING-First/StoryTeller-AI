from diffusers import StableDiffusionPipeline
import torch
import os

import warnings
warnings.filterwarnings("ignore")

class ImageGenerator:
    def __init__(self, pre_trained_model_name="Bingsu/my-korean-stable-diffusion-v1-5", lora_path="../models/lora-diffusion-weight/pytorch_lora_weights.safetensors", save_path="../fairyTale_images"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pre_trained_model_name = pre_trained_model_name
        self.lora_scale = 0.9
        self.lora_path = lora_path
        self.save_path = save_path

        self.pipeline = None

    def load_diffusion_model(self):
        print("Diffusion 모델 로딩 중......")
        print(f"Diffusion 베이스 모델: {self.pre_trained_model_name}")
        print(f"Diffusion Lora 경로: {self.lora_path}")

        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                                self.pre_trained_model_name,
                                torch_dtype=self.dtype
                            ).to(self.device)
            
            self.pipeline.load_lora_weights(self.lora_path, prefix=None)

            print("Diffusion 모델 로딩 완료")

        except Exception as e:
            print(f"Diffusion 모델 로딩 실패: {str(e)}")
            self.pipeline = None


    def generate_image(self, prompt, title):
        os.makedirs(os.path.join(self.save_path, title), exist_ok=True)

        if self.pipeline is None:
            print("모델이 로드 되지 않았습니다.")
            return False

        try:
            pipeline_output = self.pipeline(
                prompt=[prompt],
                num_inference_steps=20,
                cross_attention_kwargs={"scale": self.lora_scale},
                generator=torch.manual_seed(101)
            )

            count = len(os.listdir(os.path.join(self.save_path, title))) + 1

            pipeline_output.images[0].save(os.path.join(self.save_path, title, f"{title}_{str(count).zfill(6)}.png"))

            return True

        except Exception as e:
            print(f"이미지 생성 실패: {str(e)}")
            return False
        

if __name__ == "__main__":
    gi = ImageGenerator()
    gi.load_diffusion_model()
    gi.generate_image("명성왕후가 얼음나라 공주를 만나서 차를 마셨어요", "명성왕후")
