# TF.js 物件偵測範例 (tf-webcam)

使用 TensorFlow.js 與 YOLO 26 模型的靜態網頁範例，支援圖片上傳與物件偵測。

本專案採用 **本地端（瀏覽器端）推論**：模型與推論流程會在你的裝置上執行（透過 TensorFlow.js），不會把影像上傳到後端伺服器。

## 隱私 / 離線推論

- **推論位置**：在使用者的瀏覽器中執行（client-side）
- **資料流向**：相機影像與上傳圖片不會送到任何伺服器
- **後端 / API**：本專案**沒有**後端服務，**沒有**任何額外 API 介面（純前端靜態頁面）

## YOLO 26 簡介

YOLO 26（又稱 YOLOv26 / Ultralytics YOLO26）於 **2025 年 9 月**釋出，主打在**邊緣裝置**與**低功耗設備**上兼顧效率、準確率與可部署性。依據你提供的參考內容，其重點包含：

- **架構/訓練改動**：移除 Distribution Focal Loss（DFL）、採用端到端 **NMS-free** 推論、加入 ProgLoss 與 Small-Target-Aware Label Assignment（STAL）、引入 MuSGD 以提升收斂穩定性
- **多任務支援**：物件偵測、實例分割（segmentation）、姿態/關鍵點（pose/keypoints）、旋轉框（oriented detection）、分類（classification）
- **部署導向**：支援多種匯出/部署路徑（例如 ONNX、TensorRT、CoreML、TFLite），並可搭配 INT8 / FP16 量化以提升即時性

## 線上範例

- **Demo 網站**：`https://zxc88645.github.io/tf-webcam/`

## 部署到 GitHub Pages

1. **確認程式碼已推送到 GitHub**
   - 若尚未建立遠端儲存庫，在 GitHub 建立新 repo（例如 `tf-webcam`），然後：

   ```bash
   git remote add origin https://github.com/你的帳號/tf-webcam.git
   git branch -M main
   git push -u origin main
   ```

2. **啟用 GitHub Pages**
   - 進入 repo 的 **Settings** → **Pages**
   - **Source** 選擇 **GitHub Actions**
   - 按 **Save**

3. **等待部署**
   - 幾分鐘後，網站會出現在：
   - `https://你的帳號.github.io/tf-webcam/`

## 本機執行

使用 Vite 開發伺服器（推薦）：

```bash
npm install
npm run dev
```

然後在瀏覽器開啟終端機顯示的網址（通常是 `http://localhost:5173`）。

## Commit hook（格式化 + ESLint）

本專案使用 **Husky + lint-staged** 在 `git commit` 前自動執行：

- **Prettier**：先對已暫存（staged）的檔案做格式化修正
- **ESLint**：再對已暫存的 JS/TS 檔做 `--fix`，最後以 `--max-warnings=0` 檢查（不允許 warnings）

只要執行過一次依賴安裝，hook 就會啟用：

```bash
npm install
```

## 專案結構

- `src/` — 應用程式來源碼（React + TypeScript）
  - `app/` — UI 入口（`App.tsx`）
  - `features/` — 功能模組（camera / detection）
  - `lib/` — 推論與繪製等共用邏輯
  - `styles/` — 全域樣式（Tailwind）
- `public/` — 靜態資產（Vite build 時原樣拷貝到 `dist/`）
  - `version.json` — 部署版本資訊（由腳本更新）
  - `models/` — TensorFlow.js GraphModel（`model.json` + shards）
    - `yolo26n-pose/`
    - `yolo26n-detect/`
    - `yolo26n-seg/`
- `scripts/` — 開發/部署輔助腳本（例如產生 `public/version.json`）
