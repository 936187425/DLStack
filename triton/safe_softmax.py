import torch
import triton
import triton.language as tl

# triton.language模块中:
# program_id(axis): 返回当前程序实例沿给定axis的ID。axis必须是0,1或2
# num_programs(axis): 返回沿给定 axis 启动的程序实例数量。axis必须是0,1或2
# load(): 其中有一个cache_modifer可以设置为{"",".ca",".cg",".cv"}之一,
# 其中“.ca”表示在所有级别缓存,".cg"表示在全局级别缓存(在L2及以下缓存,而不是L1),“.cv”表示不缓存并重新获取。
# store(): 将数据张量存储到由pointer定义的内存位置.
# range: 这是一个特殊的迭代器,用于在triton.jit函数的上下文中实现与Python的range相似的语义。此外，它允许用户向编译器传递额外属性。



#triton与pytorch相比,就是可以进行算子融合,下面要介绍的是safe-softmax
#原理: 每个程序根据程序数量跨步加载输入矩阵X的一组行,对其进行归一化，并将结果写回输出Y
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    #(1) starting row of the program 
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    #(2) 使用已有线程块跨步循环遍历所有行
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
    # 在tl.range中, num_stages表示对当前的for loop进行多级流水线化,也就是在循环的一次迭代中,会加载num_stages份数据(num_stages行)


        #(3) 获取当前线程索引
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE) # 该程序所处理的偏移数据量
        input_ptrs = row_start_ptr + col_offsets # 该程序所处理的输入数据的指针

        #(4) 加载当前block的数据
        row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

        #(5) 计算当前线程的softmax输出
        row_minus_max = row - tl.max(row)
        row_exp = tl.exp(row_minus_max)
        denominator = tl.sum(row_exp)
        softmax_output = row_exp / denominator

        #(6) 写回当前线程的softmax输出
        output_start_ptr = output_ptr+row_idx * output_row_stride
        output_cols_ptr = output_start_ptr + col_offsets
        tl.store(output_cols_ptr,softmax_output,mask=col_offsets<n_cols)  

# triton.jit修饰的函数将在GPU上编译和运行。它只能访问以下内容:
# 1. Python原语
# 2. triton包中的内置函数
# 3. 此函数的参数
# 4. 其他jit编译的函数

# triton获取的device
properties = triton.runtime.driver.active.utils.get_device_properties(0)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"] #字节
WARP_SIZE = properties["warpSize"]


def softmax(x):
    n_rows,n_cols= x.shape

    #Triton的一个重要限制是每个块的元素数量必须是2的幂
    BLOCK_SIZE= triton.next_power_of_2(n_cols)

    #num_warps
    num_warps = 8

    # Number of software pipelining stages
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # allocate output
    y = torch.empty_like(x)

    # GPU计算优化中常见的kernel预热(warmup)操作,
    # 主要用于预编译CUDA Kernel并获取寄存器占用和线程占用率(Occupancy)信息,
    # 以便优化后续的实际计算性能
    kernel = softmax_kernel.warmup(y,x,x.stride(0),y.stride(0),n_rows,n_cols,BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages,num_warps=num_warps,grid=(1,))
    kernel._init_handles()
    

if __name__ == '__main__':
    print(NUM_REGS,NUM_REGS,SIZE_SMEM,WARP_SIZE) 