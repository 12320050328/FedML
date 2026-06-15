1. torch_fedavg_medmnist_lr_example.py
主要聯邦學習程式。
功能：載入 PathMNIST、切分 Client 資料、建立模型、執行 FedAvg
2. fedml_config.yaml
FedML 設定檔。
包含：Client 數量、Communication Round、Epoch、Batch Size、Learning Rate
3. run_fedavg_and_export.py
額外實作程式。
功能：執行 FedAvg、擷取每輪結果、繪製 Accuracy Curve、繪製 Loss Curve、匯出 Excel

