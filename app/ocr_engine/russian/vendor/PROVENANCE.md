# RussianDocsOCR provenance

The embedded engine is derived from
[RussianDocsOCR v1.0.1](https://github.com/protei300/RussianDocsOCR/tree/v1.0.1),
commit `8f75028a8015e91a6011f9616c56e51eccbebeab`. The upstream package and
model files are distributed under Apache License 2.0; its license is retained
as `LICENSE.upstream` in this directory.

The integration keeps only the passport-page Python path and ONNX runtime
support used by this service. Unsupported CoreML, OpenVINO, H5 and TFLite
paths, quality and anti-spoofing stages, scripts, samples, benchmark output and
Python bytecode are excluded.

The seven required ONNX model sets are byte-identical to v1.0.1. Their paths,
sizes and SHA-256 digests are pinned in
`app/ocr_engine/russian/models.lock.json`. Model binaries are not stored in
this repository. `install.sh` downloads them from the immutable upstream
commit into `storage/models/russian-docs-ocr/1.0.1/models` and verifies every
file before the Russian OCR worker starts.
