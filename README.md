# Nghe An & Ha Tinh ASR & Translation

Dự án cung cấp hệ thống nhận diện giọng nói (ASR - Automatic Speech Recognition) phương ngữ Nghệ An/Hà Tĩnh và dịch nghĩa sang tiếng Việt phổ thông sử dụng mô hình ngôn ngữ lớn (LLM) kết hợp cơ chế RAG (Retrieval-Augmented Generation) tra cứu từ điển phương ngữ địa phương.

---

## Tính năng chính

1. **Nhận diện giọng nói (ASR)**:
   - Tự động nhận diện giọng nói từ file âm thanh sử dụng mô hình fine-tuned Whisper.
   - Hỗ trợ chuyển đổi âm thanh sang định dạng chuẩn và xử lý trực tiếp bằng thư viện `transformers`.
2. **Dịch thuật phương ngữ Nghệ Tĩnh**:
   - Dịch các từ và câu nói thuộc phương ngữ Nghệ An, Hà Tĩnh sang tiếng Việt phổ thông một cách tự nhiên.
   - Kết hợp giữa khớp từ điển n-gram, khoảng cách chỉnh sửa (Levenshtein fuzzy match) và tìm kiếm ngữ nghĩa Vector DB (ChromaDB + `BAAI/bge-m3` embedding) làm ngữ cảnh gợi ý cho LLM.
   - Kết nối với API LLM (qua OpenRouter, mặc định sử dụng `deepseek/deepseek-v4-flash`).
3. **Web Interface trực quan**:
   - Giao diện frontend đơn giản để người dùng ghi âm, tải file âm thanh, nhận diện giọng nói và dịch trực tiếp.

---

## Cấu trúc thư mục dự án

```text
├── main.py                          # Điểm chạy chính (FastAPI Server)
├── asr_service.py                   # Xử lý nhận diện giọng nói bằng Whisper
├── translator.py                    # Xử lý RAG và kết nối LLM dịch phương ngữ
├── smart_normalize.py               # Chuẩn hóa nhãn văn bản
├── nghe_an_dict_qwen3_cleaned.json # Từ điển dữ liệu phương ngữ Nghệ An - Hà Tĩnh
├── metadata/                        # Thư mục chứa dữ liệu thống kê & nhãn bổ sung
│   ├── speaker_info.json
│   └── labels.json
├── static/                          # Giao diện Frontend (HTML, CSS, JS)
│   ├── index.html
│   ├── index.css
│   └── index.js
├── scripts/                         # Các helper script bổ sung
│   └── export_whisper_peft_to_web_onnx.py  # Script gộp adapter & xuất định dạng ONNX
├── requirements-onnx.txt            # Danh sách thư viện Python cần cài đặt
├── .env.example                     # File cấu hình mẫu môi trường
└── README.md                        # Tài liệu hướng dẫn sử dụng (File này)
```

---

## Hướng dẫn cài đặt và khởi chạy

### 1. Chuẩn bị môi trường
Yêu cầu Python từ 3.10 trở lên. Khuyên dùng môi trường ảo `venv`:

```powershell
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo (Windows)
.\.venv\Scripts\Activate.ps1

# Kích hoạt môi trường ảo (Linux/macOS)
source .venv/bin/activate

# Cập nhật pip và cài đặt thư viện
pip install -r requirements-onnx.txt
```

### 2. Cấu hình môi trường (`.env`)
Copy file `.env.example` thành `.env` và điền khóa API của bạn:

```powershell
cp .env.example .env
```

Mở `.env` và điền key OpenRouter của bạn để sử dụng tính năng dịch:
```env
OPENROUTER_API_KEY=sk-or-v1-... # Điền API Key của bạn tại đây
OPENROUTER_MODEL=deepseek/deepseek-v4-flash
PORT=8000
ASR_MODEL_PATH=build/onnx-web/merged/whisper-lora-nghe-tinh
DICTIONARY_PATH=nghe_an_dict_qwen3_cleaned.json
```

### 3. Chạy ứng dụng FastAPI Server

Khởi động server backend FastAPI bằng cách chạy file `main.py`:

```powershell
python main.py
```

Server sẽ mặc định chạy tại địa chỉ: `http://localhost:8000`. Bạn có thể truy cập thẳng vào đường dẫn này trên trình duyệt để sử dụng giao diện web.

---

## Hướng dẫn xuất mô hình sang ONNX chạy trên Web

Nếu muốn chạy mô hình Whisper trực tiếp trên trình duyệt client-side (sử dụng Transformers.js) thay vì gọi server inference, bạn có thể thực hiện gộp adapter PEFT và xuất ra ONNX:

```powershell
python scripts\export_whisper_peft_to_web_onnx.py --force
```

Xem thêm hướng dẫn chi tiết tại [README_ONNX_WEB.md](README_ONNX_WEB.md).

---

## Giấy phép (License)
Dự án được phân phối dưới giấy phép MIT License. Xem file [LICENSE](LICENSE) để biết thêm chi tiết.
