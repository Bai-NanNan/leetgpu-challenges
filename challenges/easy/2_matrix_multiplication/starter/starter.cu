#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    /*
        naive版本，不使用shared memory
    */
    //计算对应C中的位置
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    if (row >= M || col >= K) {
        return;
    }
    float sum = 0.0f;
    //对于C[row][col]而言，来自A的row行，B的col列
    for(int i=0; i<N; i++){
        sum += (A[row*N+i] * B[i*K+col]);
    }
    C[row*K+col] = sum;
    
}
//参考https://dlog.com.cn/posts/cuda11/matmul，开始迭代之路

// constexpr int TILE_M = 8;//一个 Block负责C的8行
// constexpr int TILE_K = 32;//一个Block负责C的32列，对应一个warp
// constexpr int TILE_N = 32;//每次沿着N维度处理32个元素
// __global__ void matrix_multiplication_kernel_v1Tile(const float* A, const float* B, float* C, int M, int N, int K){
//     /*
//         v1 Tile版本
//         一个 Block 计算 C 中一个 8 × 32 的输出块：C tile: BLOCK_M × BLOCK_K = 8 × 32
//         规约维度 N 每次处理 32 个元素：A tile: 8 × 32, B tile: 32 × 32
//     */
//     //局部位置
//     const int local_col = threadIdx.x;
//     const int local_row = threadIdx.y;
//     //在C中的全局索引
//     const int row = blockIdx.y*TILE_M + local_row;
//     const int col = blockIdx.x*TILE_K + local_col;
//     float sum = 0.0f;
//     __shared__ float A_shared[TILE_M][TILE_N];
//     __shared__ float B_shared[TILE_N][TILE_K];
//     for(int tile_start = 0; tile_start < N; tile_start += TILE_N){//沿着N维度分块
//         //Step 1：协作读取A的Tile,每个Thread读一个元素
//         const int a_col = tile_start + local_col;
//         if(row<M && a_col<N){
//             A_shared[local_row][local_col] = A[row*N + a_col];
//         }else{
//             A_shared[local_row][local_col] = 0.0f;//越界填0
//         }
//         //Step2：读取B的Tile，每个Thread读4个元素，要合并访存
//         #pragma unroll
//         for(int load_index = 0; load_index < TILE_N/TILE_M; load_index++){
//             //计算一下是几倍的差距，单个Thread就要读几个
//             const int b_local_row = local_row + load_index * TILE_M;//本次在tile内的局部坐标
//             const int b_row = tile_start + b_local_row;//映射回B中坐标
//             if(b_row < N && col < K){
//                 B_shared[b_local_row][local_col] = B[b_row*K + col];//合并访存
//             }else{
//                 B_shared[b_local_row][local_col] = 0;
//             }
//         }
//         __syncthreads();//整个Block都读完
//         //Step3：计算当前Tile
//         #pragma unroll
//         for(int n = 0; n<TILE_N; n++){
//             sum += A_shared[local_row][n]*B_shared[n][local_col];
//         }
//         __syncthreads();
//     }
//     if (row < M && col < K) {
//         C[row * K + col] = sum;
//     }
// }


constexpr int TILE_M = 64;
constexpr int TILE_K = 32;
constexpr int TILE_N = 32;
constexpr int TM = 8;//每个线程负责计算C中同一列上的 TM 个输出元素
__global__ void matrix_multiplication_kernel_v2(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K){
    /*
    Block尺寸：blockDim.x = TILE_K = 32; blockDim.y = TILE_M / TM = 8
    threadIdx.x表示负责输出的列，threadIdx.y表示负责输出的TM=8行的起点组，组的编号
    */
    //block内当前线程的输出坐标
    const int thread_col = threadIdx.x;
    const int thread_row_group = threadIdx.y;//按TM分组，组的编号
    //当前blcok负责的C Tile左上角坐标
    const int block_row_start = blockIdx.y * TILE_M;
    const int block_col_start = blockIdx.x * TILE_K;
    //当前Thread对应C的列
    const int global_col = block_col_start + thread_col;
    //当前Thread对应C的TM行的起点行
    const int local_row_start = thread_row_group * TM;
    const int global_row_start = block_row_start + local_row_start;
    /*
    当前Blcok需要缓存
        A Tile Size: TILM_M × TILE_N 64 × 32
        B Tile Size: TILE_N × TILE_K 32 × 32
    */
    __shared__ float A_shared[TILE_M][TILE_N];
    __shared__ float B_shared[TILE_N][TILE_K];
    float sum[TM] = {0.0f};
    //Thread在block的一维索引
    const int linear_tid = threadIdx.y*blockDim.x + threadIdx.x;
    const int threads_per_block = blockDim.x * blockDim.y;
    /*
    沿规约维度 N 分块,每轮处理：
        A[:, tile_start : tile_start + TILE_N]
        B[tile_start : tile_start + TILE_N, :]
    */
    for(int tile_start = 0; tile_start < N; tile_start += TILE_N){
        //加载A Tile，一个A Tile 64*32元素，一个block有32*8个，一个Thread读取8个
        #pragma unroll
        for(int idx = linear_tid; idx < TILE_M*TILE_N; idx += threads_per_block){//(64*32)/(32*8) = 8
            //以block内所有thread数量为单位跳着读，因为相邻位置有其他Thread在读取
            //映射回A中坐标
            const int local_row = idx / TILE_N;
            const int local_n = idx % TILE_N;           //因为N维度上规约，所以col改记为n
            const int global_row = block_row_start + local_row;
            const int global_n = tile_start + local_n;  //因为N维度上规约，所以col改记为n
            //读取
            if (global_row < M && global_n < N) {
                A_shared[local_row][local_n] =
                    A[global_row * N + global_n];
            } else {
                A_shared[local_row][local_n] = 0.0f;
            }
        } 
        //加载B Tile 一个B Tile 32*32个元素，一个Thread读4个
        #pragma unroll
        for(int idx = linear_tid; idx<TILE_N*TILE_K; idx += threads_per_block){
            //映射坐标
            const int local_n = idx / TILE_K;
            const int local_col = idx % TILE_K;
            const int global_n = tile_start + local_n;
            const int global_col = block_col_start + local_col;
            if (global_n < N && global_col < K) {
                B_shared[local_n][local_col] = B[global_n*K + global_col];
            }else{
                B_shared[local_n][local_col] = 0.0f;
            }
        }
        __syncthreads();
        /*
        计算，对于N维度上的规约
            每个位置从B_shared读1个元素，A_shared读TM个，更新TM个累加器，最后写回C的TM×1位置
        */
        #pragma unroll
        for(int n=0; n<TILE_N; n++){
            //局部向量积和
            const float b_value = B_shared[n][thread_col];//复用同一个B Tile中的元素，收益来源
            #pragma unroll
            for(int m=0; m<TM; m++){
                //TM个输入，对应A中TM个
                sum[m] += b_value*A_shared[local_row_start+m][n];
            }
        }
        __syncthreads();
    }
    //写回C
    #pragma unroll
    for(int m=0; m<TM; m++){
        const int global_row = global_row_start + m;
        if(global_row<M && global_col<K){
            C[global_row*K+global_col] = sum[m];
        }
    }

    

}


// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    //有点反人类，A是M×N，B是N×K
    //naive版本 Benchmark   median 870.0665 ms  min 867.1198 ms  p20 868.9116 ms  p80 922.6246 ms
    // dim3 threadsPerBlock(16, 16);
    // dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
    //                    (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    
    //V1 Tile Benchmark   median 536.7654 ms  min 532.7357 ms  p20 534.3418 ms  p80 544.2740 ms
    // dim3 threadsPerBlock(TILE_K,TILE_M);
    // dim3 blocksPerGrid(
    //     (K+TILE_K-1)/TILE_K,
    //     (M+TILE_M-1)/TILE_M
    // );
    // matrix_multiplication_kernel_v1Tile<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, M, N, K);

    //V2 1D Thread Benchmark   median 238.8346 ms  min 213.8755 ms  p20 226.1282 ms  p80 245.1370 ms
    dim3 threadsPerBlock(TILE_K,TILE_M/TM);
    dim3 blocksPerGrid(
        (K+TILE_K-1)/TILE_K,
        (M+TILE_M-1)/TILE_M
    );
    matrix_multiplication_kernel_v2<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, M, N, K);



    cudaDeviceSynchronize();
}
