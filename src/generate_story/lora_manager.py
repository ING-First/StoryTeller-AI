from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from peft import PeftModel
import torch
import gc
from typing import Dict, Optional, Union
import threading

class SharedLoRAManager:
  _instance = None
  _lock = threading.Lock()
  
  def __new__(cls):
    if cls._instance is None:
      with cls._lock:
        if cls._instance is None:
          cls._instance = super().__new__(cls)
          
    return cls._instance
  
  def __init__(self):
    if hasattr(self, 'initialized'):
      return
    
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if self.device == "cuda":
      torch.backends.cuda.matmul.allow_tf32 = True
      torch.set_float32_matmul_precision("high")
      
    self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    self.lora_paths = {
      "eval": BASE_DIR.parent / "models" / "lora_eval",
      "summary": BASE_DIR.parent / "models" / "lora_summary"
    }
    
    self.base_model = None
    self.tokenizer = None
    self.current_lora = None
    self.current_model = None
    self.loaded_loras = {}
    
    self.initialized = True
    print(f"[LoRAManager] 초기화 완료: Device={self.device}, dtype={self.dtype}")
    
  def register_lora_path(self, name: str, path: Union[str, Path]):
    self.lora_paths[name] = Path(path)
    print(f"[LoRAManager] LoRA 경로 등록: {name} -> {path}")
    
  def load_base_model(self, base_model_id: str = "kimssai/sk-a.x-4.0-light-8bit") -> bool:
    if self.base_model is not None:
      print(f"[LoRAManager] 베이스 모델 이미 로드됨: {base_model_id}")
      return True
    
    try:
      print(f"[LoRAManger] 베이스 모델 로딩 시작: {base_model_id}")
      
      self.tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code = True
      )
      
      if self.tokenizer.pad_token is None:
        self.tokenizer.pad_token = self.tokenizer.eos_token
      self.tokenizer.padding_side = "left"
      
      self.base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map='auto' if self.device == 'cuda' else None,
        torch_dtype=self.dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True
      )
      
      if self.device == 'cpu':
        self.base_model.to(self.device)
        
      self.base_model.eval()
      self.current_model = self.base_model
      
      print(f"[LoRAManager] 베이스 모델 로딩 완료")
      return True
    
    except Exception as e:
      print(f"[LoRAManager] 베이스 모델 로딩 실패")
      return False
    
  def switch_lora(self, lora_name: str, force_reload: bool = False) -> bool:
      """LoRA 어댑터 스위칭"""
      if self.base_model is None:
          raise RuntimeError("베이스 모델이 로드되지 않았습니다. load_base_model()을 먼저 호출하세요.")
      
      if self.current_lora == lora_name and not force_reload:
          print(f"[LoRAManager] {lora_name} LoRA 이미 활성화됨")
          return True
      
      # 베이스 모델 사용 요청
      if lora_name == "base" or lora_name is None:
          self.current_model = self.base_model
          self.current_lora = "base"
          print(f"[LoRAManager] 베이스 모델로 스위치")
          return True
      
      # 캐시에서 확인
      if lora_name in self.loaded_loras and not force_reload:
          self.current_model = self.loaded_loras[lora_name]
          self.current_lora = lora_name
          print(f"[LoRAManager] 캐시에서 {lora_name} LoRA 로드")
          return True
      
      # 새로운 LoRA 로드
      try:
          lora_path = self.lora_paths.get(lora_name)
          if lora_path is None:
              print(f"[LoRAManager] 알 수 없는 LoRA: {lora_name}")
              print(f"[LoRAManager] 사용 가능한 LoRA: {list(self.lora_paths.keys())}")
              return False
          
          if not lora_path.exists():
              print(f"[LoRAManager] LoRA 경로가 존재하지 않음: {lora_path}")
              return False
          
          print(f"[LoRAManager] {lora_name} LoRA 로딩 시작: {lora_path}")
          
          # 메모리 정리 (기존 current_model이 베이스가 아닌 경우)
          if self.current_model != self.base_model and self.current_lora in self.loaded_loras:
              # 캐시는 유지하되 current만 변경
              pass
          
          # 새 LoRA 로드
          lora_model = PeftModel.from_pretrained(
              self.base_model, 
              lora_path,
              torch_dtype=self.dtype
          )
          
          if self.device == "cpu":
              lora_model.to(self.device)
          lora_model.eval()
          
          # 캐시에 저장
          self.loaded_loras[lora_name] = lora_model
          self.current_model = lora_model
          self.current_lora = lora_name
          
          print(f"[LoRAManager] {lora_name} LoRA 로딩 완료")
          return True
          
      except Exception as e:
          print(f"[LoRAManager] {lora_name} LoRA 로딩 실패: {e}")
          self.current_model = self.base_model
          self.current_lora = "base"
          return False
  
  def get_current_model(self):
      """현재 활성화된 모델 반환"""
      if self.current_model is None:
          raise RuntimeError("모델이 로드되지 않았습니다.")
      return self.current_model
  
  def get_tokenizer(self):
      """토크나이저 반환"""
      if self.tokenizer is None:
          raise RuntimeError("토크나이저가 로드되지 않았습니다.")
      return self.tokenizer
  
  def get_current_lora_name(self) -> Optional[str]:
      """현재 활성화된 LoRA 이름 반환"""
      return self.current_lora
  
  def list_available_loras(self) -> list:
      """사용 가능한 LoRA 목록 반환"""
      return list(self.lora_paths.keys())
  
  def list_loaded_loras(self) -> list:
      """캐시에 로드된 LoRA 목록 반환"""
      return list(self.loaded_loras.keys())
  
  def unload_lora(self, lora_name: str) -> bool:
      """특정 LoRA를 캐시에서 제거"""
      if lora_name in self.loaded_loras:
          # 현재 사용 중인 LoRA인 경우 베이스로 스위치
          if self.current_lora == lora_name:
              self.switch_lora("base")
          
          del self.loaded_loras[lora_name]
          print(f"[LoRAManager] {lora_name} LoRA 언로드 완료")
          
          if torch.cuda.is_available():
              torch.cuda.empty_cache()
          gc.collect()
          return True
      return False
  
  def cleanup_cache(self):
      print(f"[LoRAManager] 캐시 정리 시작: {len(self.loaded_loras)}개 LoRA")
      
      for lora_name in list(self.loaded_loras.keys()):
          self.unload_lora(lora_name)
      
      if torch.cuda.is_available():
          torch.cuda.empty_cache()
      gc.collect()
      
      print(f"[LoRAManager] 캐시 정리 완료")
  
  def get_memory_info(self) -> dict:
      # 메모리 사용 정보
      info = {
          "loaded_loras": list(self.loaded_loras.keys()),
          "current_lora": self.current_lora,
          "device": self.device,
      }
      
      if torch.cuda.is_available():
          info.update({
              "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
              "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,
          })
      
      return info
    
def get_lora_manager() -> SharedLoRAManager:
    return SharedLoRAManager()


def switch_to_lora(lora_name: str) -> bool:
    manager = get_lora_manager()
    return manager.switch_lora(lora_name)


def ensure_model_loaded(base_model_id: str = "kimssai/sk-a.x-4.0-light-8bit") -> bool:
    manager = get_lora_manager()
    return manager.load_base_model(base_model_id)