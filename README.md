# TF.js 物件偵測範例 (tf-webcam)

使用 TensorFlow.js 與 YOLO 模型的靜態網頁範例，支援圖片上傳與物件偵測。

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
