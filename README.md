# TF.js 物件偵測範例 (tf-webcam)

使用 TensorFlow.js 與 YOLO 模型的靜態網頁範例，支援圖片上傳與物件偵測。

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
   - **Source** 選擇 **Deploy from a branch**
   - **Branch** 選 `main`，**Folder** 選 **/public**
   - 按 **Save**

3. **等待部署**
   - 幾分鐘後，網站會出現在：
   - `https://你的帳號.github.io/tf-webcam/`

## 本機執行

用任意靜態伺服器開啟 `public/` 即可，例如：

```bash
npx serve public
# 或
cd public && python -m http.server 8000
```

然後在瀏覽器開啟 `http://localhost:3000`（或 8000）。

## 專案結構

- `public/` — 可直接部署的靜態網站（GitHub Pages 指向此資料夾）
  - `index.html` — 主頁面
  - `app.js` — TF.js 推理與相機/圖片流程
  - `styles.css` — UI 樣式
  - `version.json` — 部署版本資訊（由腳本更新）
  - `models/` — TensorFlow.js GraphModel（`model.json` + shards）
    - `yolo26n-pose/`
    - `yolo26n-detect/`
- `scripts/` — 開發/部署輔助腳本
