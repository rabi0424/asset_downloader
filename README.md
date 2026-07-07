# Asset Downloader (ComfyUI custom node)

HuggingFaceまたはCivitaiのURLから、チェックポイント/LoRA等のモデルファイルをダウンロードし、
ComfyUIの対応するモデルフォルダ（`models/checkpoints`, `models/loras`など）に保存するノードです。

## インストール

このリポジトリを `ComfyUI/custom_nodes/` 配下にクローンし、ComfyUIを再起動してください。

```
cd ComfyUI/custom_nodes
git clone <this-repo>
pip install -r asset_downloader/requirements.txt
```

## ノード: "Download Model/LoRA (HF/Civitai)"

- **url** (必須): HuggingFaceまたはCivitaiのURL
  - HuggingFace: リポジトリURL（例 `https://huggingface.co/{repo}`）、または
    ファイル直リンク（`.../resolve/main/{file}` / `.../blob/main/{file}`）
  - Civitai: モデルページURL（`https://civitai.com/models/{id}?modelVersionId={id}`）、
    または直接ダウンロードURL（`https://civitai.com/api/download/models/{id}`）
- **save_type** (必須): 保存先フォルダ種別
  (`checkpoints` / `loras` / `vae` / `text_encoders` / `controlnet` / `embeddings` / `upscale_models` / `unet` / `clip` / `clip_vision`)
  - `text_encoders`は新しめのComfyUIが持つフォルダ種別です。未対応の古いバージョンでは自動的に`clip`フォルダにフォールバックします。
- **filename** (任意): 保存ファイル名を明示指定。HuggingFaceのリポジトリURLのみを指定した場合、
  リポジトリ内にモデルファイルが複数あるときはこの入力でファイルパスを指定する必要があります。
- **overwrite** (任意, デフォルト `False`): `True`にすると常に再ダウンロードします。
- **max_connections** (任意, デフォルト `4`, 1〜8): サーバーがHTTP Range(部分取得)に対応しており、
  かつファイルサイズが32MiB以上の場合に、指定した本数の並列コネクションで分割ダウンロードします。
  Rangeに対応していないサーバーだった場合や分割ダウンロードが失敗した場合は、自動的に単一コネクションでの
  ダウンロードにフォールバックします。

出力はダウンロード済みファイルのフルパス（STRING）です。既存の `Load Checkpoint` / `LoraLoader`
などのノードには直接接続できないため、ファイルパスをログ確認や他のカスタムノードでの利用に使ってください。

## 認証（非公開/レート制限対策）

APIキーは環境変数から読み込みます（ワークフローJSONには保存されません）。

- `CIVITAI_API_TOKEN`: Civitaiの非公開モデルやダウンロード制限のあるモデル用
- `HF_TOKEN`: HuggingFaceのgated/非公開リポジトリ用

ComfyUIを起動する前にシェルでエクスポートするか、起動スクリプトに設定してください。

```
export CIVITAI_API_TOKEN=xxxxx
export HF_TOKEN=hf_xxxxx
```

## 既存ファイルの扱い

保存先に同名ファイルが既に存在する場合、リモート側のメタデータ（Civitaiは`SHA256`、
HuggingFaceはファイルサイズ）と比較し、完全に一致すればダウンロードをスキップします。
一致しない場合、または `overwrite=True` の場合は再ダウンロードして上書きします。
