# experiments

## parameters

| gpu   | block size (fine tuned size) | block numbers (not for all kernels) |
| ----- | ---------------------------- | ----------------------------------- |
| mx150 | 256                          | 1ULL << 20                          |
| V100  | 256                          | 1ULL << 22                          |

## mx150

```
kernel v1: 82.537 ms, 12.116 GB/s
kernel v2: 48.973 ms, 20.419 GB/s
kernel v3: 30.002 ms, 33.331 GB/s
kernel v4: 30.304 ms, 32.999 GB/s
kernel v5: 30.006 ms, 33.326 GB/s
kernel v6: 29.199 ms, 34.248 GB/s
```

### V100

```
kernel v1: 10.782 ms, 371.000 GB/s
kernel v2: 6.856 ms, 583.454 GB/s
kernel v3: 4.864 ms, 822.318 GB/s
kernel v4: 4.870 ms, 821.343 GB/s
kernel v5: 4.855 ms, 823.824 GB/s
kernel v6: 4.802 ms, 832.903 GB/s
```

V100 memory bandwidth 900 GB/s

kernel v6 达到了 92.5% 的吞吐量。

### key optimization points

- global memory throughput, each block load two blocks of global memory.
- block size has been fined-tuned to be 256,  the size can be 64, 128, 256, 512, 1024.

- [ ] warp reduce 效果不好，需要分析下原因
- [ ] 测试发现loop unroll 没有效果，需要反汇编分析下

