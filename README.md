# MUSE-C web demo deploy手順

## 0. 前提

- `chord_model.pt`（タグ条件コードモデル）
- `mel_finetune.pt`（Phase 3メロディ分離モデル、語彙225）
- Python環境（torch / onnx / onnxruntime）

## 1. ONNX化

```bash
py web/export_onnx.py --chord chord_model.pt --piano mel_finetune.pt --out web/models
```

`web/models/chord_model.onnx`（約1MB）と`web/models/piano_model.onnx`（約40MB）ができる。
parityチェック付き（失敗したら止まる）。

## 2. GitHub Pages公開

```bash
cd web
git init && git add index.html app.js models/ && git commit -m "musec demo"
gh repo create musec-demo --public --source=. --push
# → Settings → Pages → Deploy from branch → main / (root)
```

## 3. 動作確認（iPad）

1. 公開URLをiPad Safariで開く
2. EPはまず`wasm`で確認→`webgpu`を試す
3. 既知の制約：
   - iPad SafariのWebGPUは不安定報告あり。動かなければwasmに戻す
   - 初回はシェーダコンパイルでもたつく
   - NEは使えない（ブラウザの仕様）。NE狙いはネイティブ版で

## 4. クレジット表示（必須）

ページ下部に以下を記載（POP909 CC BY 4.0）：

> 学習データ: POP909 Dataset (Z. Wang et al., ISMIR 2020, CC BY 4.0)
