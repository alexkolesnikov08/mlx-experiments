# MLX Experiments

Учебный проект — эксперименты с кастомными Metal-ядрами для ускорения инференса на Apple Silicon через MLX.

- **MLX port** depthwise MNIST model: PyTorch → MLX (channel-last)
- **Fused Metal kernel**: depthwise conv 3×3 + BatchNorm + ReLU в один launch
- **Benchmark**: сравниваем PyTorch MPS, MLX baseline, fused kernel

Run:
```bash
python train_mnist.py          # train (PyTorch MPS)
python benchmark_all.py         # benchmark all backends
python plot_benchmark.py        # update comparison plots
```
