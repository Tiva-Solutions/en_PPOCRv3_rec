import os

# Đường dẫn tới file labels.txt
label_file = "./plates/label.txt"

# Thư mục chứa ảnh
image_folder = "./plates/image/"

# Đọc nội dung của labels.txt
with open(label_file, "r") as file:
    lines = file.readlines()

# Khởi tạo danh sách để lưu lại các dòng hợp lệ
valid_lines = []

# Xử lý từng dòng trong labels.txt
for line in lines:
    line = line.strip()
    if not line:
        continue
    
    # Tách path và label
    parts = line.split()
    if len(parts) != 2:
        continue
    image_path, label = parts
    
    # Kiểm tra điều kiện:
    # 1. label có độ dài >= 8
    # 2. Ký tự thứ 2 không phải chữ cái
    # 3. Hai chữ số đầu trong label < 11
    if len(label) >= 8 and not label[1].isalpha():
        try:
            if int(label[:2]) > 10:
                valid_lines.append(line)
            else:
                raise ValueError
        except ValueError:
            # Nếu hai ký tự đầu không phải số hợp lệ, hoặc >= 11 thì xóa ảnh
            full_image_path = os.path.join(image_folder, os.path.basename(image_path))
            if os.path.exists(full_image_path):
                os.remove(full_image_path)
                print(f"Đã xóa ảnh (label không hợp lệ): {full_image_path}")

    else:
        # Xóa ảnh nếu không thỏa mãn các điều kiện khác
        full_image_path = os.path.join(image_folder, os.path.basename(image_path))
        if os.path.exists(full_image_path):
            os.remove(full_image_path)
            print(f"Đã xóa ảnh: {full_image_path}")

# Ghi lại các dòng hợp lệ vào labels.txt
with open(label_file, "w") as file:
    file.write("\n".join(valid_lines))

print("Hoàn tất xử lý!")
