Note:

## 1D texture

1. **tex1Dfetch** -> linear memory

> tex1Dfetch() only works with non-normalized coordinates, 
> so only the border and clamp addressing modes are supported.
> It does not perform any texture filtering.

也就是说，1D线性内存没什么用！

2. **tex1D** -> cudaArray


