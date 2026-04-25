import os

# Prepend PyTorch CUDA DLL path so onnxruntime-gpu can find cublasLt64_12.dll etc.
# Required on Windows when CUDA Toolkit is not installed system-wide.
_cuda_dll_path = os.getenv("CUDA_DLL_PATH", "")
if _cuda_dll_path and _cuda_dll_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _cuda_dll_path + os.pathsep + os.environ.get("PATH", "")

import src.api.routes
from src.api.app import app
from src.config import SERVER_HOST, SERVER_PORT

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
