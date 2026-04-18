# TensorFlow Lite Model Optimization Demo

End-to-end demo comparing a standard Keras CNN against its TFLite and FP16-quantized TFLite counterparts on CIFAR-10 — measuring the trade-off between model size and accuracy for edge deployment.

## Results

| Model | Accuracy | Size | Size Reduction |
|---|---|---|---|
| Base Keras | 81.26% | 37.36 MB | — |
| TFLite (FP32) | 81.26% | 12.42 MB | 66.8% |
| TFLite (FP16 quantized) | 81.27% | 6.22 MB | **83.4%** |

FP16 quantization cut the model to roughly **one-sixth the original size** with no measurable accuracy loss on the CIFAR-10 test set (10,000 images).

## What's in this repo

- `Tensorflowlite_code_demo.ipynb` — end-to-end notebook: data loading, model definition, training, conversion, quantization, and evaluation
- `cifar10_cnn_model.keras` — trained Keras base model (37 MB)
- `cifar10_cnn_model.tflite` — unquantized TFLite (12 MB)
- `cifar10_cnn_quant_model.tflite` — FP16-quantized TFLite (6.2 MB)

## Model architecture

A 3-block CNN with 3,256,906 parameters:

| Block | Layers | Output |
|---|---|---|
| 1 | Conv(64, 5×5) → BN → Conv(64, 3×3) → BN → MaxPool → Dropout(0.25) | 16×16×64 |
| 2 | Conv(128) → BN → Conv(128) → BN → MaxPool → Dropout(0.3) | 8×8×128 |
| 3 | Conv(256) → BN → Conv(256) → BN → MaxPool → Dropout(0.4) | 4×4×256 |
| Head | Flatten → Dense(512) → BN → Dropout(0.5) → Dense(10, softmax) | 10 |

Trained for 10 epochs with Adam + exponential LR decay (`0.001 × 0.9^(step/10000)`). Data augmentation via `ImageDataGenerator` (rotation ±15°, translation ±10%, horizontal flip).

## Reproducing the results

```bash
pip install tensorflow numpy pandas matplotlib
jupyter notebook Tensorflowlite_code_demo.ipynb
```

Run all cells. The notebook:

1. Loads CIFAR-10 and normalizes to `[0, 1]`
2. Trains the CNN for 10 epochs
3. Saves the Keras model and reports its size
4. Converts to TFLite (FP32), evaluates with the TFLite interpreter, reports size
5. Re-converts with `Optimize.DEFAULT` + `float16` supported types, evaluates, reports size
6. Compares all three models side-by-side

## Key takeaways

- **FP16 quantization is effectively lossless for this model** — the 0.01% accuracy change is well within test-set noise.
- **Raw TFLite conversion (no quantization) already gives 67% size reduction** by dropping training-only overhead (optimizer state, graph metadata, variable wrappers).
- Full INT8 quantization (not done here) would shrink further but requires a representative calibration dataset and typically costs 0.5–2% accuracy.

## Potential next steps

- INT8 full-integer quantization with representative dataset calibration
- Latency benchmarks (CPU, mobile CPU, Coral Edge TPU)
- Post-training vs. quantization-aware training comparison
