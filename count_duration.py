import os
import wave
import contextlib
import sys

# Đảm bảo in ra tiếng Việt không bị lỗi font trên Windows Console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def count_audio_duration(folder_path):
    if not os.path.exists(folder_path):
        print(f"Lỗi: Thư mục '{folder_path}' không tồn tại!")
        return

    total_seconds = 0.0
    wav_count = 0
    other_count = 0

    print(f"Đang quét thư mục: {folder_path}...\n")

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isdir(file_path):
            continue

        if file_name.lower().endswith(".wav"):
            try:
                with contextlib.closing(wave.open(file_path, 'r')) as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = frames / float(rate)
                    total_seconds += duration
                    wav_count += 1
            except Exception as e:
                print(f"Lỗi khi đọc file {file_name}: {e}")
        else:
            # Ghi nhận các định dạng khác nếu có
            if file_name.lower().endswith(('.mp4', '.m4a', '.mp3')):
                other_count += 1

    print("================ KẾT QUẢ ĐẾM TIMING ================")
    print(f"Tổng số file WAV đo được: {wav_count} files")
    if other_count > 0:
        print(f"Số file định dạng khác (chưa đếm thời lượng): {other_count} files")
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    print(f"Tổng thời lượng: {total_seconds:.2f} giây")
    print(f"Quy đổi: {hours} giờ {minutes} phút {seconds:.2f} giây (~{total_seconds / 3600.0:.2f} giờ)")
    print("====================================================")

if __name__ == "__main__":
    # Đường dẫn mặc định tới thư mục speaker_01
    default_dir = os.path.join("processed_audio", "speaker_01")
    count_audio_duration(default_dir)
