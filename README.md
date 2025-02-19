# AI Delta

## Model Download
1. Install huggingface-cli
```pip install huggingface-hub```
2. Login Hugging Face
```huggingface-cli login```
3. Access model
https://huggingface.co/meta-llama/Llama-3.2-3B
4. Download model
```huggingface-cli download meta-llama/Llama-3.2-3B --include "original/*" --local-dir Llama-3.2-3B```