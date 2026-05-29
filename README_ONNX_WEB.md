# Whisper ONNX Web

Mục tiêu: merge hai PEFT adapter `whisper-lora-nghe-tinh` và `whisper-dora-nghe-tinh` vào base `vinai/phowhisper-base`, export sang ONNX, rồi cho browser chạy trực tiếp bằng Transformers.js thay vì gọi server inference.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-onnx.txt
```

## Export cả 2 model

```powershell
python scripts\export_whisper_peft_to_web_onnx.py --force
```

Output sẽ nằm ở:

```text
models/whisper-lora-nghe-tinh/
models/whisper-dora-nghe-tinh/
```

Mỗi model cần có layout kiểu:

```text
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
preprocessor_config.json
onnx/
  encoder_model.onnx
  decoder_model_merged.onnx
  encoder_model_int8.onnx
  decoder_model_merged_int8.onnx
```

## Chạy web local

```powershell
python -m http.server 5173 -d web
```

Mở:

```text
http://localhost:5173
```

## Ghi chú

- Script mặc định quantize dynamic int8 để giảm dung lượng tải trên web.
- Nếu cần chất lượng tốt hơn, chọn dtype `fp32` trong web hoặc export với `--quantize none`.
- Browser vẫn phải tải model ONNX lần đầu, nên `whisper-base` có thể khá nặng. Sau lần đầu, browser thường cache file model.
- Nếu export báo thiếu `decoder_model_merged.onnx`, nâng `optimum-onnx` rồi chạy lại:

```powershell
pip install --upgrade "optimum-onnx[onnxruntime]"
python scripts\export_whisper_peft_to_web_onnx.py --force
```
