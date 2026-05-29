import os
import json
import re
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def smart_normalize(label_file_path):
    backup_path = label_file_path + ".bak"
    if not os.path.exists(backup_path):
        print("Lỗi: Không tìm thấy file backup gốc để làm chuẩn!")
        return

    # Luôn đọc từ backup gốc để thực hiện sửa đổi chính xác
    with open(backup_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(label_file_path)))
    
    updated_count = 0

    for item in data:
        audio_path = item.get("audio_path")
        raw_audio_path = item.get("raw_audio_path")
        
        # 1. Chuẩn hóa raw_audio_path theo file thực tế tồn tại trên đĩa
        if raw_audio_path:
            full_path_01 = os.path.join(base_dir, raw_audio_path)
            # Thử đổi speaker_01 thành speaker_1 trong tên file (giữ nguyên folder speaker_01)
            raw_audio_path_1 = re.sub(r"speaker_0([1-9])/speaker_0\1_", r"speaker_0\1/speaker_\1_", raw_audio_path)
            full_path_1 = os.path.join(base_dir, raw_audio_path_1)
            
            if os.path.exists(full_path_1):
                item["raw_audio_path"] = raw_audio_path_1
                # Nếu file là speaker_1 thì id và speaker_id cũng nên chuyển sang speaker_1
                item["id"] = re.sub(r"speaker_0([1-9])_", r"speaker_\1_", item.get("id", ""))
                item["speaker_id"] = re.sub(r"speaker_0([1-9])$", r"speaker_\1", item.get("speaker_id", ""))
                updated_count += 1
            elif os.path.exists(full_path_01):
                # Giữ nguyên bản gốc speaker_01
                item["raw_audio_path"] = raw_audio_path
            else:
                # Nếu cả hai đều không tồn tại, thử tìm xem file nào thực sự có trong folder đó
                dir_name = os.path.dirname(full_path_01)
                if os.path.exists(dir_name):
                    base_name = os.path.basename(raw_audio_path)
                    alt_name = os.path.basename(raw_audio_path_1)
                    # Quét xem có file nào tương đương không
                    files_in_dir = os.listdir(dir_name)
                    if alt_name in files_in_dir:
                        item["raw_audio_path"] = raw_audio_path_1
                        item["id"] = re.sub(r"speaker_0([1-9])_", r"speaker_\1_", item.get("id", ""))
                        item["speaker_id"] = re.sub(r"speaker_0([1-9])$", r"speaker_\1", item.get("speaker_id", ""))
                        updated_count += 1
        
        # 2. Chuẩn hóa audio_path (processed_audio)
        if audio_path:
            # Thử đổi speaker_01 thành speaker_1 trong tên file
            audio_path_1 = re.sub(r"speaker_0([1-9])/speaker_0\1_", r"speaker_0\1/speaker_\1_", audio_path)
            full_path_01 = os.path.join(base_dir, audio_path)
            full_path_1 = os.path.join(base_dir, audio_path_1)
            
            if os.path.exists(full_path_1):
                item["audio_path"] = audio_path_1
            elif os.path.exists(full_path_01):
                item["audio_path"] = audio_path

    # Lưu lại file labels.json đã được chuẩn hóa thông minh
    with open(label_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Chuẩn hóa thông minh hoàn tất!")
    print(f"Đã cập nhật khớp thực tế {updated_count} mẫu âm thanh.")

if __name__ == "__main__":
    label_path = os.path.join("metadata", "labels.json")
    smart_normalize(label_path)
