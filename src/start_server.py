from huggingface_hub import login
import os
import subprocess
import threading
import time

# BitsAndBytes CUDA 문제 해결을 위한 환경변수 설정
os.environ['BNB_CUDA_VERSION'] = '121'
os.environ['CUDA_HOME'] = '/usr/local/cuda'
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = f"/usr/local/cuda/lib64:{current_ld_path}"

def setup_huggingface():
    """HuggingFace 로그인"""
    hf_token = os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN')
    if hf_token:
        try:
            login(hf_token)
            print("HuggingFace 로그인 성공")
        except Exception as e:
            print(f"HuggingFace 로그인 실패: {e}")
    else:
        print("HuggingFace 토큰이 설정되지 않았습니다")

def print_output(process, name):
    """프로세스 출력을 실시간으로 표시"""
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(f"[{name}] {output.strip()}", flush=True)

def start_servers():
    print("🚀 서버 실행 시작")
    setup_huggingface()
    
    # AI 서버 실행
    print("AI 서버 실행 중... (포트 8000)")
    ai_process = subprocess.Popen(
        ["uvicorn", "AI_main:app", "--host=0.0.0.0", "--port=8000"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=0
    )
    ai_thread = threading.Thread(target=print_output, args=(ai_process, "AI"))
    ai_thread.daemon = True
    ai_thread.start()
    time.sleep(3)
    
    # Backend 서버 실행
    print("Backend 서버 실행 중... (포트 8001)")
    backend_process = subprocess.Popen(
        ["uvicorn", "backend_main:app", "--host=0.0.0.0", "--port=8001"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=0
    )
    backend_thread = threading.Thread(target=print_output, args=(backend_process, "Backend"))
    backend_thread.daemon = True
    backend_thread.start()
    time.sleep(3)
    
    print("모든 서버가 실행 중입니다! (AI:8000, Backend:8001)")
    
    try:
        ai_process.wait()
        backend_process.wait()
    except KeyboardInterrupt:
        print("서버 종료 중...")
        ai_process.terminate()
        backend_process.terminate()
        print("서버가 종료되었습니다.")

if __name__ == "__main__":
  start_servers()