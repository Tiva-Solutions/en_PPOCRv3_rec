import os
import shutil
import random
import yaml

def split_labels_and_prepare_dataset(label_file, image_dir, output_dir, train_ratio=0.8):
    # Đọc nội dung file label.txt
    with open(label_file, "r") as f:
        lines = f.readlines()

    # Shuffle dữ liệu để đảm bảo ngẫu nhiên
    random.shuffle(lines)

    # Chia dữ liệu thành train và test
    num_samples = len(lines)
    train_split = int(train_ratio * num_samples)

    train_lines = lines[:train_split]
    test_lines = lines[train_split:]

    # Tạo thư mục output
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Tạo file rec_gt_train.txt và rec_gt_test.txt
    train_label_file = os.path.join(output_dir, "rec_gt_train.txt")
    test_label_file = os.path.join(output_dir, "rec_gt_test.txt")

    with open(train_label_file, "w") as f_train, open(test_label_file, "w") as f_test:
        def process_lines(lines, folder, output_file):
            for line in lines:
                image_path, label = line.strip().split()
                # label = f'"{label}"'
                image_name = os.path.basename(image_path)  # Lấy tên file ảnh (ví dụ: 0001.jpg)
                src_path = os.path.join(image_dir, image_name)  # Đường dẫn gốc từ plates/image

                # Sao chép ảnh vào thư mục train hoặc test
                dest_path = os.path.join(folder, image_name)
                if os.path.exists(src_path):
                    shutil.copy(src_path, dest_path)
                    # Ghi đường dẫn mới và label vào file output
                    output_file.write(f"./{os.path.basename(folder)}/{image_name}\t{label}\n")
                else:
                    print(f"Lỗi: Ảnh {src_path} không tồn tại!")

        process_lines(train_lines, train_dir, f_train)
        process_lines(test_lines, test_dir, f_test)

    # Tạo file config
    config_file = os.path.join(output_dir, "en_PP-OCRv3_rec.yml")
    create_config(config_file, train_label_file, test_label_file)
    print(f"Tập dữ liệu đã được chia thành công tại {output_dir}")
    print(f"- File train labels: {train_label_file}")
    print(f"- File test labels: {test_label_file}")
    print(f"- Ảnh train: {train_dir}")
    print(f"- Ảnh test: {test_dir}")
    print(f"- File config: {config_file}")


import yaml

def create_config(config_path, train_label_file, test_label_file):
    config = {
        "Global": {
            "debug": False,
            "use_gpu": True,
            "epoch_num": 500,
            "log_smooth_window": 20,
            "print_batch_step": 10,
            "save_model_dir": "/kaggle/working/output/models_finetune/en_PP-OCRv3_rec_ft",
            "save_epoch_step": 250,
            "eval_batch_step": [0, 150],
            "cal_metric_during_train": True,
            "pretrained_model": "/kaggle/working/en_PP-OCRv3_rec_train/best_accuracy.pdparams",
            "checkpoints": None,
            "save_inference_dir": "/kaggle/working/output/models_inference/best_accuracy",
            "use_visualdl": False,
            "infer_img": "/kaggle/input/license-plates-vietnamese/dataset/test/76F101980_20241221_081914.jpg",
            "character_dict_path": "/kaggle/working/PaddleOCR/ppocr/utils/en_dict.txt",
            "max_text_length": 25,
            "infer_mode": False,
            "use_space_char": True,
            "distributed": True,
            "save_res_path": "/kaggle/working/output/rec/predicts_ppocrv3.txt"
        },
        "Optimizer": {
            "name": "Adam",
            "beta1": 0.9,
            "beta2": 0.999,
            "lr": {
                "name": "Cosine",
                "learning_rate": 0.005,
                "warmup_epoch": 5
            },
            "regularizer": {
                "name": "L2",
                "factor": 3.0e-05
            }
        },
        "Architecture": {
            "model_type": "rec",
            "algorithm": "SVTR",
            "Transform": None,
            "Backbone": {
                "name": "MobileNetV1Enhance",
                "scale": 0.5,
                "last_conv_stride": [1, 2],
                "last_pool_type": "avg"
            },
            "Head": {
                "name": "MultiHead",
                "head_list": [
                    {
                        "CTCHead": {
                            "Neck": {
                                "name": "svtr",
                                "dims": 64,
                                "depth": 2,
                                "hidden_dims": 120,
                                "use_guide": True
                            },
                            "Head": {
                                "fc_decay": 0.00001
                            }
                        }
                    },
                    {
                        "SARHead": {
                            "enc_dim": 512,
                            "max_text_length": 25
                        }
                    }
                ]
            }
        },
        "Loss": {
            "name": "MultiLoss",
            "loss_config_list": [
                {"CTCLoss": None},
                {"SARLoss": None}
            ]
        },
        "PostProcess": {
            "name": "CTCLabelDecode"
        },
        "Metric": {
            "name": "RecMetric",
            "main_indicator": "acc",
            "ignore_space": False
        },
        "Train": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": "/kaggle/input/license-plates-vietnamese/dataset",
                "ext_op_transform_idx": 1,
                "label_file_list": [train_label_file],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"RecConAug": {"prob": 0.5, "ext_data_num": 2, "image_shape": [48, 320, 3], "max_text_length": 25}},
                    {"RecAug": None},
                    {"MultiLabelEncode": None},
                    {"RecResizeImg": {"image_shape": [3, 48, 320]}},
                    {"KeepKeys": {"keep_keys": ["image", "label_ctc", "label_sar", "length", "valid_ratio"]}}
                ]
            },
            "loader": {
                "shuffle": True,
                "batch_size_per_card": 128,
                "drop_last": True,
                "num_workers": 4
            }
        },
        "Eval": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": "/kaggle/input/license-plates-vietnamese/dataset",
                "label_file_list": [test_label_file],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"MultiLabelEncode": None},
                    {"RecResizeImg": {"image_shape": [3, 48, 320]}},
                    {"KeepKeys": {"keep_keys": ["image", "label_ctc", "label_sar", "length", "valid_ratio"]}}
                ]
            },
            "loader": {
                "shuffle": False,
                "drop_last": False,
                "batch_size_per_card": 128,
                "num_workers": 4
            }
        },
        "wandb": {
            "project": "OCR_with_Paddle"
        }
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)



# Gọi hàm
split_labels_and_prepare_dataset(
    label_file="./plates/label.txt",  # Đường dẫn tới file label.txt
    image_dir="./plates/image",      # Thư mục chứa ảnh
    output_dir="./dataset_1",          # Thư mục đầu ra
    train_ratio=0.8                  # Tỉ lệ train-test
)
