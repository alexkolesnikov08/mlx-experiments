# MLX Experiments

Эксперименты с кастомными Metal-ядрами для ускорения инференса на Apple Silicon через MLX.

- **MLX port** depthwise MNIST model: PyTorch → MLX (channel-last)
- **Fused Metal kernel**: depthwise conv 3×3 + BatchNorm + ReLU в один launch
- **Benchmark**: сравниваем PyTorch MPS, MLX baseline, fused kernel
- **Hardware**: переехало с M1 Air (8 GB, 8 GPU-ядер) на **M1 Pro (16 GPU-ядер, 32 GB, 200 GB/s)** — см. `conclusions/benchmark_results.md §8` для честного разбора «железо vs ПО vs качество измерений»

Run:
```bash
python train_mnist.py          # train (PyTorch MPS)
python benchmark_all.py         # benchmark all backends
python plot_benchmark.py        # update comparison plots
python model_profile.py         # FLOPs / AI / roofline (GPU_SPECS в начале файла)
```
Данные: `output/per_trial.npz` (M1 Pro), архив `output/per_trial_m1_air_8gb.npz` (M1 Air).
