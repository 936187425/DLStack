import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,y_ptr,output_ptr,n_elements,BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(0) #我们使用1D launch网格，因此axis是0
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE) #数据偏移量,此时offsets是一个tensor,所以此时是看成是一个1D的tensor
    mask = offsets < n_elements #这个元素数是指的是x_ptr对应的tensor里的数据量,mask的size (BLOCK_SIZE) boolean
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

# kernel的函数封装
def add(x: torch.Tensor, y: torch.Tensor):
    # 我们需要预先分配好输出
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda 
    n_elements = x.numel()

    # 在这种情况下,我们使用一个1D网格，其大小是块的数量
    grid = lambda meta: (triton.cdiv(n_elements,meta['BLOCK_SIZE']),)
    add_kernel[grid](x,y,output,n_elements,BLOCK_SIZE=1024)
    return output



if __name__=='__main__':
    torch.manual_seed(0)
    size=98432
    
    x = torch.rand(size,device='cuda')
    y = torch.rand(size,device='cuda')
    
    output_torch = x + y
    output_triton= add(x,y)
    print(output_torch)
    print(output_triton)
    print(f'在torch和triton之间的最大差异是 '
          f'{torch.max(torch.abs(output_torch - output_triton))}')