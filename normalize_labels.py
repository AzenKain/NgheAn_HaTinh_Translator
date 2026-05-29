import os
import json
import re
import sys

# Đảm bảo in ra tiếng Việt không bị lỗi font trên Windows Console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def normalize_labels(label_file_path):
    # Khôi phục từ file backup nếu có để đảm bảo chạy lại từ đầu chuẩn xác
    backup_path = label_file_path + ".bak"
    if os.path.exists(backup_path):
        print(f"Khôi phục dữ liệu gốc từ backup: {backup_path}")
        with open(backup_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # Nếu chưa có backup thì tạo mới
        if not os.path.exists(label_file_path):
            print(f"Error: File {label_file_path} không tồn tại!")
            return
        with open(label_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Đã tạo file backup tại: {backup_path}")

    # Thực hiện thay thế:
    # 1. id: speaker_0X_... -> speaker_X_...
    # 2. speaker_id: speaker_0X -> speaker_X
    # 3. audio_path / raw_audio_path: speaker_0X/speaker_0X_... -> speaker_0X/speaker_X_... (thư mục vẫn là 01, file là 1)
    
    modified_count = 0
    for item in data:
        old_id = item.get("id")
        
        # 1. Cập nhật ID
        if item.get("id"):
            item["id"] = re.sub(r"speaker_0([1-9])_", r"speaker_\1_", item["id"])
            
        # 2. Cập nhật speaker_id
        if item.get("speaker_id"):
            item["speaker_id"] = re.sub(r"speaker_0([1-9])$", r"speaker_\1", item["speaker_id"])
            
        # 3. Cập nhật audio_path & raw_audio_path (thư mục giữ nguyên speaker_0X, tên file đổi thành speaker_X)
        if item.get("audio_path"):
            item["audio_path"] = re.sub(r"speaker_0([1-9])/speaker_0\1_", r"speaker_0\1/speaker_\1_", item["audio_path"])
            
        if item.get("raw_audio_path"):
            item["raw_audio_path"] = re.sub(r"speaker_0([1-9])/speaker_0\1_", r"speaker_0\1/speaker_\1_", item["raw_audio_path"])
            
        if old_id != item["id"]:
            modified_count += 1

    # Lưu lại file labels.json đã được chuẩn hóa
    with open(label_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Đã chuẩn hóa thành công!")
    print(f"Số lượng bản ghi được cập nhật: {modified_count}/{len(data)}")

if __name__ == "__main__":
    label_path = os.path.join("metadata", "labels.json")
    normalize_labels(label_path)
