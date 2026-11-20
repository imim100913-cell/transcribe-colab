# transcribe-colab

Google Colab の GPU（T4 想定）を使って、Google Drive 上の音声・動画ファイルを高速に文字起こしするためのプロジェクトです。  
既存のローカル運用（`kotoba-tech/kotoba-whisper-v2.0-faster`）を置き換えるのではなく、長尺録音向けの高速版として追加することを目的としています。出力形式は後続エージェント（Muse）が読み取りやすい `.txt`（全文）＋ `[mm:ss]` タイムスタンプ付き `.md` に揃えています。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imim100913-cell/transcribe-colab/blob/main/transcribe_colab.ipynb)

---

## Colab での使い方（4ステップ）

1. **ノートブックを開く**  
   上の **Open In Colab** バッジ（または `transcribe_colab.ipynb`）を Google Colab で開きます。
2. **GPU ランタイムを選択する**  
   メニューから **「ランタイム ＞ ランタイムのタイプを変更 ＞ ハードウェア アクセラレータ: T4 GPU」** を選びます。
3. **冒頭の設定セルを編集する**  
   最初のコードセルで `INPUT_DIR`（Google Drive 内の音声フォルダ）、`LANGUAGE`（`"ja"` / `"en"` / `"auto"`）、`MODEL`（既定 `"auto"`）、`OVERWRITE`（再処理時は `True`）を設定します。
4. **上から順にセルを実行する**  
   Drive マウントの許可画面が出たら承認します。完了後、各音声ファイルと同じ階層の `transcribed/` フォルダに `<元ファイル名>.txt` と `<元ファイル名>.md` が生成されます。

---

## 対応入力と出力形式

- **入力対応拡張子**: `.mp3`, `.m4a`, `.wav`, `.mp4`（大文字・小文字不問、`INPUT_DIR` 配下を再帰的に探索）
- **除外ディレクトリ**: `transcribed/` および `.` で始まる隠しディレクトリ（`.ipynb_checkpoints` 等）
- **出力先**: 各音声ファイルと同じ階層の `transcribed/` フォルダ
  - `<stem>.txt`: セグメントを改行区切りで連結した全文プレーンテキスト
  - `<stem>.md`: `# <stem>` 見出しと `- [mm:ss] <テキスト>` 形式のタイムスタンプ付き一覧

出力例（`.md`）:

```markdown
# meeting_sample

- [00:00] 本日のミーティングを開始します。
- [01:05] 最初の議題について確認します。
```

---

## 設計判断と既定値

迷った箇所は **「安全側・単純側・再実行可能側」** を基準に以下を選択しています。

| 項目 | 決定した既定値 | 理由 |
| --- | --- | --- |
| 日本語モデル (`LANGUAGE="ja"`) | `kotoba-tech/kotoba-whisper-v2.0-faster` | 現行ローカル運用と同一モデル。Hugging Face 上での実在および `faster-whisper` での読み込みを確認済み。 |
| 英語・自動判定モデル (`LANGUAGE="en"` / `"auto"`) | `Systran/faster-whisper-large-v3` | `faster-whisper` 開発元 Systran 公式の CTranslate2 変換モデルで 100 言語対応（`distil-large-v3` は英語専用のため日英混在の `auto` に不適）。実在と読み込みを確認済み。速度優先時は `MODEL` に `mobiuslabsgmbh/faster-whisper-large-v3-turbo` や英語専用の `Systran/faster-distil-whisper-large-v3` も指定可能。 |
| GPU 精度フォールバック | `float16` 優先 → 失敗時 `int8` 自動切替 | T4 GPU では `float16` が高速。初期化に失敗した場合のみ `cuda` 上で `int8` にフォールバックし、CUDA 自体が無い環境（ローカル検証等）では警告を出して `cpu` / `int8` で動作する。 |
| `chunk_length` | `kotoba` 系のみ既定 `15`、その他は `None` (30s) | `kotoba-tech/kotoba-whisper-v2.0-faster` 公式モデルカードの推奨値（`chunk_length=15`）に準拠。 |
| `condition_on_previous_text` | `False` | 長尺録音で Whisper 特有の同一フレーズ反復（ハルシネーションループ）を防ぐ安全側設定。Kotoba 公式例とも一致。 |
| `vad_filter` | `True` | Silero VAD により無音区間の誤認識（「ごめん」等の幻聴）と不要な推論時間を削減。 |
| タイムスタンプ `[mm:ss]` | 分を切り捨てで 2 桁以上ゼロ埋め（60分以上は `[75:03]`） | 仕様の `[mm:ss]` 形式を厳密に維持しつつ 1 時間超の録音でも桁あふれしない単純な形式を採用。 |
| 再実行・再開（スキップ判定） | `.txt` と `.md` の **両方** が存在する場合のみスキップ | 書き込み途中の強制終了でも片側欠損時は再処理される。さらに一時ファイル経由の `os.replace`（原子的書き込み）で破損ファイルの残存を防止。 |
| エラー分離 | 1 ファイル失敗しても残りを継続し、最後に失敗一覧を表示 | バッチ処理中に破損ファイルが 1 件あっても他のファイルの処理が止まらないようにするため。 |
| ノートブックとロジック分離 | `transcribe_colab.ipynb` から `src/transcribe.py` を import | Colab で単体オープンされた場合は自動で本リポジトリを `/content/transcribe-colab` にクローンして `src.transcribe` を読み込む。 |

---

## リポジトリ構成

```text
transcribe-colab/
├── transcribe_colab.ipynb  # Google Colab 実行用ノートブック
├── src/
│   ├── __init__.py
│   └── transcribe.py       # 文字起こし・モデル解決・ファイル探索の本体ロジック
├── tests/
│   ├── test_transcribe.py  # 純粋関数・フォールバック・再開ロジックの単体テスト
│   └── test_notebook.py    # ノートブックの nbformat 妥当性テスト
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ローカルでのテスト実行

```bash
pip install -e . pytest nbformat
pytest -v
```

---

## 未解決事項・手動確認チェックリスト

本リポジトリの作成環境には **NVIDIA GPU（T4）および実際の音声録音データが存在しない** ため、以下は Colab 実機での手動確認が必要です（未確認のまま確認済みとはしていません）。

### 手動確認チェックリスト（Google Colab T4 GPU）

- [ ] **Colab 起動・Drive マウント**: 「Open In Colab」から `transcribe_colab.ipynb` を開き、T4 GPU ランタイムで Google Drive がマウントできること
- [ ] **日本語実音声の文字起こし (`LANGUAGE="ja"`)**: `kotoba-tech/kotoba-whisper-v2.0-faster` が `cuda` / `float16` でロードされ、実時間より大幅に高速（例: 10〜30倍速程度）に処理できること
- [ ] **英語／自動判定の文字起こし (`LANGUAGE="en"` / `"auto"`)**: `Systran/faster-whisper-large-v3` が T4 の VRAM（16 GB）内で安定動作し、英語・自動判定が正しく出力されること
- [ ] **全対応形式の読込**: `.mp3` / `.m4a` / `.wav` / `.mp4`（動画からの音声トラック抽出含む）およびサブフォルダ再帰処理がエラーなく完了すること
- [ ] **中断・再開の検証**: 複数ファイル処理中にランタイムを一度停止し、再実行時に処理済みファイルが `SKIP` されて未処理ファイルから再開すること（および `OVERWRITE=True` で上書き再生成されること）
- [ ] **Muse 連携確認**: 生成された `<name>.txt` と `<name>.md` を Muse に渡し、既存ローカル運用と同等に解析・処理できること

### 既知の制約・今後の改善候補

- **話者分離（Diarization）は未実装**: 単純性・T4 での安定性を優先し、本バージョンでは単一話者/タイムスタンプ一覧のみを出力します。
- **Google Drive の I/O 遅延**: 大量の小ファイルや巨大 `.mp4` を Drive マウント経由で直接読む場合、Colab の `/content/` へ一時コピーする方が速いケースがあります（現時点では単純側・Drive 容量追加不要の直接読み込みを採用）。
