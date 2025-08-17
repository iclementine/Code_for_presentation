## Small Reduction size

在 reduction 维度的 size 小的情况下，使用 2d tile, `(TILE_BATCH, TILE_REDUCTION)`, 这样可以同时处理多行。一个简单的 Heuristic 是预设每个 thread 读取 4 个 float, 计算一下需要多少行。

reduction/chained_reduction_2d_small_reduction_size.py

```python
VECTOR_SIZE = 128 / item_size
TILE_REDUCTION = triton.next_power_of_2(reduction_size)
TILE_BATCH = triton.cdiv(num_warps * 32 * VECTOR_SIZE)
```

如果 batch size 很大，酌情考虑 GSL.

```python
num_tasks = triton.cdiv(batch_size, TILE_BATCH)
MAX_GRID_X = (2 ** 31) - 1
grid_x = min(MAX_GRID_X, num_tasks)
USE_GSL_KERNEL = num_tasks > MAX_GRID_X
grid = (grid_x, 1, 1)
```

如果 batch 也很小，那可能就到了需要更细的任务划分的时候了。

## Large Reduction size

这种情况下使用按 tile 串行处理的 reduction, 一般情况下不需要 TILE_BATCH, 因为 reduction size 就足够了。这个时候使用 1D TILE. 一般情况下使用

```python
TILE_REDUCTION = 4096
num_warps = 4
```

就可以了，调成其他值影响基本不大。

### 如果 batch_size 足够大

那么 grid size 也会很大，可以计算一下 wave efficiency.

```python
num_waves = triton.cdiv(batch_size, num_sms)
valid_waves = batch_size / num_sms
efficiency = valid_waves / num_waves
```

只要 batch size 够大，基本可以很接近 100%. 这种情况下使用 chained_reduction_1d.

而且由于 batch_size 大的缘故，可以考虑 GSL.

### 如果 batch_size 比较小

到不一定要比 num_sms 还小，其实只要小到一定程度，又不是 num_sms 的整数倍，就可能会带来 tail effect, 使性能出现波动。这也是 split-k 和 stream-k 开始起作用的时候。

reduction/parallel_reduction.py

atomic 方式更新即可。


