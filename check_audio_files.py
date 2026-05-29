import os
import json
import sys

# Đảm bảo in ra tiếng Việt không bị lỗi font trên Windows Console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def check_missing_audio(label_file_path):
    if not os.path.exists(label_file_path):
        print(f"Error: File {label_file_path} không tồn tại!")
        return

    with open(label_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Base directory is the parent directory of 'metadata' folder (which is the project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(label_file_path)))
    
    missing_audio = []
    missing_raw = []
    
    for item in data:
        audio_path = item.get("audio_path")
        raw_audio_path = item.get("raw_audio_path")
        item_id = item.get("id", "Unknown ID")
        
        if audio_path:
            full_audio_path = os.path.join(base_dir, audio_path)
            if not os.path.exists(full_audio_path):
                missing_audio.append((item_id, audio_path))
        else:
            missing_audio.append((item_id, "N/A (Missing field in JSON)"))
            
        if raw_audio_path:
            full_raw_path = os.path.join(base_dir, raw_audio_path)
            if not os.path.exists(full_raw_path):
                missing_raw.append((item_id, raw_audio_path))
        else:
            missing_raw.append((item_id, "N/A (Missing field in JSON)"))

    print(f"=== KẾT QUẢ KIỂM TRA ===")
    print(f"Tổng số mẫu trong JSON: {len(data)}")
    print(f"Thiếu audio_path (processed): {len(missing_audio)} file")
    print(f"Thiếu raw_audio_path (raw): {len(missing_raw)} file")
    
    if missing_audio:
        print("\n--- Chi tiết các file thiếu ở audio_path ---")
        for item_id, path in missing_audio[:50]:
            print(f"- ID: {item_id} | Path: {path}")
        if len(missing_audio) > 50:
            print(f"... và {len(missing_audio) - 50} file khác.")

    if missing_raw:
        print("\n--- Chi tiết các file thiếu ở raw_audio_path ---")
        for item_id, path in missing_raw[:50]:
            print(f"- ID: {item_id} | Path: {path}")
        if len(missing_raw) > 50:
            print(f"... và {len(missing_raw) - 50} file khác.")

if __name__ == "__main__":
    # Đường dẫn tới labels.json
    label_path = os.path.join("metadata", "labels.json")
    check_missing_audio(label_path)
