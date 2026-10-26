# 胸腺癌 vs 胸腺瘤 · 13 程序转录组签名在线预测工具

## 本地运行
```bash
cd /home/ubuntu/NSCLC/thymoma
/home/ubuntu/miniconda3/envs/sc/bin/streamlit run webapp/app.py
```
浏览器打开 http://localhost:8501

## 输入格式
CSV 文件，**行 = 基因（human gene symbol，大写），列 = 样本**，值为表达量。
- 支持 RNA-seq counts / TPM / 芯片信号（算法按样本内排名归一化，跨平台稳健）。
- 支持多列多样本同时预测。
- 首列非数值时自动当作基因名列；否则取最后一列当基因名。

示例（`test_input.csv`）：
```
gene,Sample1,Sample2
EPCAM,7.1,8.2
KRT8,6.5,7.0
...
```

## 模型文件
`model.joblib` 由 `scripts/17_export_model.py` 生成，包含 13 基因集 + 标准化参数 + 逻辑回归系数。

## 部署
推送到 GitHub 后，在 Streamlit Community Cloud / HuggingFace Spaces 新建 app，指向本目录即可（自动读 requirements.txt）。
