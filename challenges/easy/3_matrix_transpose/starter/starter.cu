#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    /*
    计算本Thread对应原矩阵的行列
    这里认为矩阵是input[row][col]
    CUDA二维矩阵认为x是水平方向，y是垂直方向变化
    */
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row<rows && col<cols){
        output[col*rows+row] = input[row*cols+col];
        //相邻线程读取相邻元素，属于合并访问；但是相邻线程写入地址间隔为 rows，属于跨步访问，大量访存事务
    }
    //Benchmark   median 2.6218 ms  min 2.5816 ms  p20 2.5975 ms  p80 2.6569 ms
}

constexpr int TILE_DIM   = 32;//一个block负责转置一个 32×32 的矩阵小块
constexpr int BLOCK_ROWS = 32;//一个block在行方向发起8个thread，也就是一个block启动32*8个thread

__global__ void matrix_transpose_kernel_sharedmemory(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int rows, 
    int cols) {
    /*
        先在shared memory完成转置再写回
    */
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];//每行额外填充一个 float，避免bankconflict
    //计算本Thread在输入矩阵中的坐标
    int input_col = blockIdx.x * TILE_DIM + threadIdx.x;
    int input_row = blockIdx.y * TILE_DIM + threadIdx.y;
    /*
        一个 Block 使用 32 × 8 = 256 个线程，
        每个线程处理 4 个元素。
        整个 Block 合作加载一个 32 × 32 的 tile。
    */
    #pragma unroll//NVCC指令，尝试展开循环
    for(int offset = 0; offset < TILE_DIM; offset += BLOCK_ROWS){
        int row = input_row + offset;
        //offset是行方向的偏移量
        if(input_col<cols && row<rows){
            tile[threadIdx.y+offset][threadIdx.x] = input[row*cols+input_col];
            //input_col和threadIdx.x一样连续，所以还是保持了合并访存
        }
    }
    __syncthreads();
    /*
        转置后：
        原矩阵中的 blockIdx.y 变成输出矩阵的列方向；
        原矩阵中的 blockIdx.x 变成输出矩阵的行方向。
    */
    int output_col = blockIdx.y * TILE_DIM + threadIdx.x;
    int output_row = blockIdx.x * TILE_DIM + threadIdx.y;
    #pragma unroll
    for(int offset = 0; offset < TILE_DIM; offset += BLOCK_ROWS){
        int row = output_row + offset;
        if(output_col < rows && row < cols){
            output[row * rows + output_col] = tile[threadIdx.x][threadIdx.y+offset];
        }
    }
    //Benchmark   median 2.7927 ms  min 2.7847 ms  p20 2.7895 ms  p80 2.7999 ms
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    // dim3 threadsPerBlock(16, 16);//每个Block对应原矩阵的一个16*16的块
    // dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    //    (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    
    dim3 threads_per_block(TILE_DIM,BLOCK_ROWS);
    dim3 blocks_per_grid(
        (cols+TILE_DIM-1)/TILE_DIM,
        (rows+TILE_DIM-1)/TILE_DIM
    );
    matrix_transpose_kernel_sharedmemory<<<blocks_per_grid, threads_per_block>>>(input, output, rows, cols);
    
    cudaDeviceSynchronize();
}
