#include <cuda_runtime.h>
#include <cstdint>

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
// __global__ void matrix_multiplication_kernel_v1Tile(
//     const float* A, 
//     const float* B, 
//     float* C, 
//     int M, int N, int K){
//     /*
//         v1 Tile版本
//         一个 Block 计算 C 中一个 8 × 32 的输出块：C tile: BLOCK_M × BLOCK_K = 8 × 32
//         归约维度 N 每次处理 32 个元素：A tile: 8 × 32, B tile: 32 × 32
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


// constexpr int TILE_M = 64;
// constexpr int TILE_K = 32;
// constexpr int TILE_N = 32;
// constexpr int TM = 8;//每个线程负责计算C中同一列上的 TM 个输出元素
// __global__ void matrix_multiplication_kernel_v2(
//     const float* __restrict__ A,
//     const float* __restrict__ B,
//     float* __restrict__ C,
//     int M,
//     int N,
//     int K){
//     /*
//     Block尺寸：blockDim.x = TILE_K = 32; blockDim.y = TILE_M / TM = 8
//     threadIdx.x表示负责输出的列，threadIdx.y表示负责输出的TM=8行的起点组，组的编号
//     */
//     //block内当前线程的输出坐标
//     const int thread_col = threadIdx.x;
//     const int thread_row_group = threadIdx.y;//按TM分组，组的编号
//     //当前blcok负责的C Tile左上角坐标
//     const int block_row_start = blockIdx.y * TILE_M;
//     const int block_col_start = blockIdx.x * TILE_K;
//     //当前Thread对应C的列
//     const int global_col = block_col_start + thread_col;
//     //当前Thread对应C的TM行的起点行
//     const int local_row_start = thread_row_group * TM;
//     const int global_row_start = block_row_start + local_row_start;
//     /*
//     当前Blcok需要缓存
//         A Tile Size: TILE_M × TILE_N 64 × 32
//         B Tile Size: TILE_N × TILE_K 32 × 32
//     */
//     __shared__ float A_shared[TILE_M][TILE_N];
//     __shared__ float B_shared[TILE_N][TILE_K];
//     float sum[TM] = {0.0f};
//     //Thread在block的一维索引
//     const int linear_tid = threadIdx.y*blockDim.x + threadIdx.x;
//     const int threads_per_block = blockDim.x * blockDim.y;
//     /*
//     沿归约维度 N 分块,每轮处理：
//         A[:, tile_start : tile_start + TILE_N]
//         B[tile_start : tile_start + TILE_N, :]
//     */
//     for(int tile_start = 0; tile_start < N; tile_start += TILE_N){
//         //加载A Tile，一个A Tile 64*32元素，一个block有32*8个，一个Thread读取8个
//         #pragma unroll
//         for(int idx = linear_tid; idx < TILE_M*TILE_N; idx += threads_per_block){//(64*32)/(32*8) = 8
//             //以block内所有thread数量为单位跳着读，因为相邻位置有其他Thread在读取
//             //映射回A中坐标
//             const int local_row = idx / TILE_N;
//             const int local_n = idx % TILE_N;           //因为N维度上归约，所以col改记为n
//             const int global_row = block_row_start + local_row;
//             const int global_n = tile_start + local_n;  //因为N维度上归约，所以col改记为n
//             //读取
//             if (global_row < M && global_n < N) {
//                 A_shared[local_row][local_n] =
//                     A[global_row * N + global_n];
//             } else {
//                 A_shared[local_row][local_n] = 0.0f;
//             }
//         } 
//         //加载B Tile 一个B Tile 32*32个元素，一个Thread读4个
//         #pragma unroll
//         for(int idx = linear_tid; idx<TILE_N*TILE_K; idx += threads_per_block){
//             //映射坐标
//             const int local_n = idx / TILE_K;
//             const int local_col = idx % TILE_K;
//             const int global_n = tile_start + local_n;
//             const int global_col = block_col_start + local_col;
//             if (global_n < N && global_col < K) {
//                 B_shared[local_n][local_col] = B[global_n*K + global_col];
//             }else{
//                 B_shared[local_n][local_col] = 0.0f;
//             }
//         }
//         __syncthreads();
//         /*
//         计算，对于N维度上的归约
//             每个位置从B_shared读1个元素，A_shared读TM个，更新TM个累加器，最后写回C的TM×1位置
//         */
//         #pragma unroll
//         for(int n=0; n<TILE_N; n++){
//             //局部向量积和
//             const float b_value = B_shared[n][thread_col];//复用同一个B Tile中的元素，收益来源
//             #pragma unroll
//             for(int m=0; m<TM; m++){
//                 //TM个输入，对应A中TM个
//                 sum[m] += b_value*A_shared[local_row_start+m][n];
//             }
//         }
//         __syncthreads();
//     }
//     //写回C
//     #pragma unroll
//     for(int m=0; m<TM; m++){
//         const int global_row = global_row_start + m;
//         if(global_row<M && global_col<K){
//             C[global_row*K+global_col] = sum[m];
//         }
//     }

    

// }


/*
一个Block负责C中TILE_M*TILE_K的输出,Block沿着N方向的归约步长为16;
一个Thread负责THREAD_TILE_M*THREAD_TILE_K元素
一个Block中THREADS_M*THREADS_K个Thread
*/
// constexpr int TILE_M = 128;  
// constexpr int TILE_K = 128; 
// constexpr int TILE_N = 32;
// constexpr int THREAD_M = 8;
// constexpr int THREAD_K = 8;
// constexpr int THREADS_K = TILE_K / THREAD_K;  
// constexpr int THREADS_M = TILE_M / THREAD_M;
// constexpr int THREADS_PER_BLOCK = THREADS_M * THREADS_K;
// __global__ void matrix_multiplication_kernel_v3(
//     const float* __restrict__ A,
//     const float* __restrict__ B,
//     float* __restrict__ C,
//     int M,
//     int N,
//     int K){
//     //AB块放到shared memory，block共享；Thread输出块留在寄存器
//     __shared__ float A_shared[TILE_M][TILE_N];
//     __shared__ float B_shared[TILE_N][TILE_M];
//     float accum[THREAD_M][THREAD_K];
//     #pragma unroll
//     for (int i = 0; i < THREAD_M; ++i) {
//         #pragma unroll
//         for (int j = 0; j < THREAD_K; ++j) {
//             accum[i][j] = 0.0f;
//         }
//     }
//     //当前block对应C中的输出块的起始行列
//     const int block_row = blockIdx.y * TILE_M;
//     const int block_col = blockIdx.x * TILE_K;
//     //当前Thread在block的一维编号
//     const int tx = threadIdx.x;
//     const int ty = threadIdx.y;
//     const int tid = ty*THREADS_K+tx;
//     //当前Thread输出块在Block输出块中起始输出行列
//     const int thread_row = ty * THREAD_M;
//     const int thread_col = tx * THREAD_K;
//     //沿着归约维度N推进, n是当前归约块在n维度的起始索引
//     for(int n=0; n<N; n+=TILE_N){
//         //读取A Tile，每个线程读(64*64)/256=4，相邻线程读取相邻元素合并访存
//         for(int i=tid; i<TILE_M*TILE_N; i+= THREADS_PER_BLOCK){
//             //A Tile中局部坐标
//             const int local_row = i/TILE_N;
//             const int local_n   = i%TILE_N;
//             //在A的全局坐标
//             const int global_row    = block_row + local_row;
//             const int global_n      = n + local_n;
//             if(global_row<M && global_n < N){
//                 A_shared[local_row][local_n] = A[global_row*N + global_n];
//             }else{
//                 A_shared[local_row][local_n] = 0.0f;
//             }
//         }
//         //读取B Tile
//         for(int i=tid; i<TILE_N*TILE_K; i+= THREADS_PER_BLOCK){
//             //B Tile中的局部坐标
//             const int local_n   = i/TILE_K;
//             const int local_col = i%TILE_K;
//             //B中全局坐标
//             const int global_n =    n + local_n;
//             const int global_col =  block_col + local_col;
//             if(global_n<N && global_col<K){
//                 B_shared[local_n][local_col] = B[global_n*K + global_col];
//             }else{
//                 B_shared[local_n][local_col] = 0.0f;
//             }
//         }
//         __syncthreads();
//         //在当前归约块，或者说归约步进行计算
//         #pragma unroll
//         for(int n=0; n<TILE_N; n++){
//             //读取A、B，当前线程负责4行，归约块内维度N为1，所以A B要4*1,1*4个值;这里的n和之前的意义不同
//             float A_fragment[THREAD_M];
//             float B_fragment[THREAD_K];
//             #pragma unroll
//             for(int i=0; i<THREAD_M; i++){
//                 A_fragment[i] = A_shared[thread_row+i][n];
//             }
//             #pragma unroll
//             for(int j=0; j<THREAD_K; j++){
//                 B_fragment[j] = B_shared[n][thread_col+j];
//             }
//             //在当前归约块内，沿着归约维度N进行计算
//             #pragma unroll
//             for(int i=0; i<THREAD_M; i++){
//                 #pragma unroll
//                 for(int j=0; j<THREAD_K; j++){
//                     accum[i][j] += A_fragment[i]*B_fragment[j];
//                 }
//             }
//         }
//         __syncthreads();//Block内所有Thread计算结束
//     }
//     //将4*4结果写回
//     #pragma unroll
//     for(int i=0; i<THREAD_M; i++){
//         //计算对应C全局坐标的行
//         const int global_row = block_row + thread_row + i;
//         #pragma unroll
//         for(int j=0; j<THREAD_K; j++){
//             //计算C中全局坐标列
//             const int global_col = block_col + thread_col + j;
//             if(global_row<M && global_col<K){
//                 C[global_row*K + global_col] = accum[i][j];
//             }
//         }
//     }
// }



/*
加入向量化访存，shared memory布局优化，双缓冲
Block Tile：128 × 128
K Tile：       8
Thread Tile：  8 × 8
Block：       16 × 16 = 256 Threads = 8 Warps
*/
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
/*A 转置写入 Shared Memory 后，每一行添加 4 个 float 的 Padding。
 As 的实际布局为：
     As[BK][BM + A_PAD]
132 × sizeof(float) = 528 Bytes，仍然是 16 字节的整数倍，
因此后续可以安全使用 float4 读取连续的 A 元素。*/
constexpr int A_PAD = 4;
constexpr int THREADS_X = BN / TN;
constexpr int THREADS_Y = BM / TM;
constexpr int THREADS_PER_BLOCK = THREADS_X * THREADS_Y;
static_assert(
    THREADS_X == 16 && THREADS_Y == 16,
    "Expected a 16 x 16 thread block."
);
static_assert(
    THREADS_PER_BLOCK == 256,
    "Expected 256 threads per block."
);
// A Tile 和 B Tile 都包含 1024 个 float。
// 256 个线程每线程加载一个 float4，恰好覆盖整个 Tile。
static_assert(
    BM * BK == THREADS_PER_BLOCK * 4,
    "Each thread must load one float4 from A."
);
static_assert(
    BK * BN == THREADS_PER_BLOCK * 4,
    "Each thread must load one float4 from B."
);
//float4读取辅助函数
template <bool UseVectorizedIO>
__device__ __forceinline__ float4 load_row_float4(
    const float* __restrict__ matrix,
    int row,
    int col4,
    int rows,
    int cols
) {
    // 越界位置补零，使边界 Tile 仍能执行相同的计算流程。
    float4 value = make_float4(
        0.0f,
        0.0f,
        0.0f,
        0.0f
    );

    if (UseVectorizedIO) {
        /*
        向量化路径由 solve() 保证：

        1. matrix 基地址满足 16 字节对齐；
        2. 行跨度 cols 是 4 的倍数；
        3. col4 是 4 的倍数。

        因此，只要 row 和 col4 没有越界，
        就能安全读取完整的 float4。
        */
        if (row < rows && col4 < cols) {
            const float* ptr =
                matrix
                + static_cast<size_t>(row) * cols
                + col4;

            value =
                *reinterpret_cast<const float4*>(ptr);
        }
    } else {
        /*
        任意尺寸下的安全标量路径。

        当 cols 不是 4 的倍数时，最后一次逻辑 float4 读取
        可能只有 1～3 个有效元素，因此逐元素判断。
        */
        if (row < rows) {
            const size_t row_offset =
                static_cast<size_t>(row) * cols;

            if (col4 + 0 < cols) {
                value.x =
                    matrix[row_offset + col4 + 0];
            }

            if (col4 + 1 < cols) {
                value.y =
                    matrix[row_offset + col4 + 1];
            }

            if (col4 + 2 < cols) {
                value.z =
                    matrix[row_offset + col4 + 2];
            }

            if (col4 + 3 < cols) {
                value.w =
                    matrix[row_offset + col4 + 3];
            }
        }
    }

    return value;
}
/*
寄存器写Shared Memory，从global memory读完后使用
A 的全局Tile原始布局:   A_tile[BM][BK]
A 转置写入Shared Memory:    As[BK][BM + A_PAD]
B 保持 row-major Shared Memory 布局:Bs[BK][BN]
*/
__device__ __forceinline__ void commit_tile_to_shared(
    float* As, float* Bs,
    const float4 a_value, const float4 b_value,
    int a_row, int a_k4, int b_k, int b_col4
){
    constexpr int AS_STRIDE = BM + A_PAD;
    /*
    从A中横向读取A[a_row][a_k4 + 0 ... a_k4 + 3]，
    转置写入As[a_k4 + 0 ... a_k4 + 3][a_row]，
    这样计算阶段在固定 kk 下，线程需要的多个 A 行元素会连续排列。
    */
    As[(a_k4 + 0) * AS_STRIDE + a_row] = a_value.x;
    As[(a_k4 + 1) * AS_STRIDE + a_row] = a_value.y;
    As[(a_k4 + 2) * AS_STRIDE + a_row] = a_value.z;
    As[(a_k4 + 3) * AS_STRIDE + a_row] = a_value.w;
    //B不转置
    *reinterpret_cast<float4*>(Bs + b_k * BN + b_col4) = b_value;
}

template <bool UseVectorizedIO>
__global__ __launch_bounds__(THREADS_PER_BLOCK) //告知编译器block中线程数量固定
void matrix_multiplication_kernel_v4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K){
        __shared__ __align__(16) float As[2][BK][BM+A_PAD];//16Byte内存对齐
        __shared__ __align__(16) float Bs[2][BK][BM];
        const int tid = threadIdx.y*blockDim.x + threadIdx.x;
        //当前Block对应C中Tile左上角
        const int block_row = blockIdx.y * BM;
        const int block_col = blockIdx.x * BN;
        //当前Thread的TM*TN微块在Block的Tile的左上角
        const int thread_row = threadIdx.y*TM;
        const int thread_col = threadIdx.x*TN;
        //Block内Thread加载映射,分配在本Block对应的A、BTile中的相对起始点
        const int a_row = tid/(BK/4);
        const int a_k4  = (tid % (BK/4))*4;
        const int b_k   = tid/(BN/4);
        const int b_col4= (tid % (BN/4))*4;
        //Register累加器
        float accum[TM][TN];
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                accum[i][j] = 0.0f;
            }
        }
        /*
        流水线启动阶段，加载第0个K Tile到buffer
        “每个 Thread 只读一个 float4”
        是当前 Tile 尺寸、vector 宽度和 Block 线程数三者设计成刚好匹配后的结果
        */
        float4 prefetched_a = load_row_float4<UseVectorizedIO>(A, block_row+a_row, a_k4, M, K);
        float4 prefetched_b = load_row_float4<UseVectorizedIO>(B, b_k, block_col+b_col4, K, N);
        commit_tile_to_shared(
            &As[0][0][0], &Bs[0][0][0],
            prefetched_a, prefetched_b,
            a_row, a_k4,
            b_k, b_col4
        );
        __syncthreads();
        const int tile_count = (K + BK - 1) / BK;//计算归约维度K上要切几块
        //归约循环
        for(int tile = 0; tile<tile_count; tile++){
            const int current_buffer = tile & 1;//位运算判断奇偶，即使用哪个缓冲区
            const int next_buffer = current_buffer ^ 1;
            const bool has_next_tile = (tile + 1) < tile_count;//如果本次是最后一块，就不启动预读
            //Step 1 启动寄存器预取;串行编程,但是没有数据依赖关系所以是编译器大概率能覆盖一定Global Memory延迟
            if(has_next_tile){
                const int next_k_base = (tile+1)*BK;
                prefetched_a = load_row_float4<UseVectorizedIO>(
                    A, 
                    block_row + a_row, 
                    next_k_base + a_k4, 
                    M, K);
                prefetched_b = load_row_float4<UseVectorizedIO>(
                    B, 
                    next_k_base + b_k, 
                    block_col+b_col4, 
                    K, N);
                //不写shared memory是为了避免引入同步，从而避免开销
            }
            //Step 2 使用缓存好的部分计算8*8
            #pragma unroll
            for(int kk = 0; kk<BK; kk++){
                //目前buffer中的A已经转置存放,这样写也是参数耦合的结果
                const float4 a_lo = *reinterpret_cast<const float4*>(&As[current_buffer][kk][thread_row+0]);
                const float4 a_hi = *reinterpret_cast<const float4*>(&As[current_buffer][kk][thread_row+4]);
                const float4 b_lo = *reinterpret_cast<const float4*>(&Bs[current_buffer][kk][thread_col+0]);
                const float4 b_hi = *reinterpret_cast<const float4*>(&Bs[current_buffer][kk][thread_col+4]);
                //展开，方便循环写法，相信编译器会优化
                const float a_frag[TM] = {a_lo.x,a_lo.y,a_lo.z,a_lo.w,a_hi.x,a_hi.y,a_hi.z,a_hi.w};
                const float b_frag[TN] = {b_lo.x,b_lo.y,b_lo.z,b_lo.w,b_hi.x,b_hi.y,b_hi.z,b_hi.w};
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        accum[i][j] += a_frag[i] * b_frag[j];
                    }
                }
            }
            //Step 3 将预取Tile写入shared memory，这里才写buffer，推迟同步，提高重叠度
            if (has_next_tile) {
                commit_tile_to_shared(
                    &As[next_buffer][0][0], &Bs[next_buffer][0][0],
                    prefetched_a, prefetched_b,
                    a_row, a_k4, b_k, b_col4);
                /*
                这一次同步同时保证：
                1. 所有线程已完成当前 Tile 的计算；
                2. 所有线程已完成下一 Tile 的 Shared Memory 写入；
                3. 下一轮可以安全交换两套 Buffer。
                */
                __syncthreads();
            }
        }
        //结果写回C
        #pragma unroll
        for(int i = 0;i<TM;i++){
            const int global_row = block_row + thread_row + i;
            const int global_col = block_col + thread_col;
            if (global_row < M){
                const size_t output_offset = static_cast<size_t>(global_row) * N + global_col;//该行的线性地址
                if (UseVectorizedIO){
                    if (global_col + 3 < N) {
                        const float4 output_lo =make_float4(accum[i][0],accum[i][1],accum[i][2],accum[i][3]);
                        *reinterpret_cast<float4*>(C + output_offset) = output_lo;
                    }
                    if (global_col + 7 < N) {
                        const float4 output_lo =make_float4(accum[i][4],accum[i][5],accum[i][6],accum[i][7]);
                        *reinterpret_cast<float4*>(C + output_offset + 4) = output_lo;
                    }
                }else{
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        if (global_col + j < N) C[output_offset + j] =accum[i][j];
                    }
                }
            }
        }
    }



// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    //有点反人类，A是M×N，B是N×K
    //naive版本 Benchmark   median 870.0665 ms  min 867.1198 ms  p20 868.9116 ms  p80 922.6246 ms
    //Benchmark   median 208.2250 ms  min 206.8214 ms  p20 207.6800 ms  p80 210.8952 ms(RTX 5070)
    // dim3 threadsPerBlock(16, 16);
    // dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
    //                    (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    
    //V1 Tile Benchmark   median 536.7654 ms  min 532.7357 ms  p20 534.3418 ms  p80 544.2740 ms
    //Benchmark   median 169.5916 ms  min 169.5440 ms  p20 169.5859 ms  p80 169.5984 ms(RTX 5070)
    // dim3 threadsPerBlock(TILE_K,TILE_M);
    // dim3 blocksPerGrid(
    //     (K+TILE_K-1)/TILE_K,
    //     (M+TILE_M-1)/TILE_M
    // );
    // matrix_multiplication_kernel_v1Tile<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, M, N, K);

    //V2 1D Thread Benchmark   median 238.8346 ms  min 213.8755 ms  p20 226.1282 ms  p80 245.1370 ms
    //Benchmark   median 53.5555 ms  min 53.4644 ms  p20 53.5307 ms  p80 53.5715 ms(RTX 5070)
    // dim3 threadsPerBlock(TILE_K,TILE_M/TM);
    // dim3 blocksPerGrid(
    //     (K+TILE_K-1)/TILE_K,
    //     (M+TILE_M-1)/TILE_M
    // );
    // matrix_multiplication_kernel_v2<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, M, N, K);

    //V3 2D Thread Benchmark   median 26.2666 ms  min 26.0221 ms  p20 26.1997 ms  p80 26.3475 ms(RTX 5070)其实超参很重要，调一调速度能快不少
    // dim3 threadsPerBlock(THREADS_K,THREADS_M);
    // dim3 blocksPerGrid(
    //     (K+TILE_K-1)/TILE_K,
    //     (M+TILE_M-1)/TILE_M
    // );
    // matrix_multiplication_kernel_v3<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, M, N, K);


    /*
    V4 加入向量化访存，shared memory布局优化，双缓冲
    交换N和K，符合大众习惯
    Benchmark   median 24.1452 ms  min 23.9954 ms  p20 24.1010 ms  p80 24.2676 ms(RTX 5070)
    */
    const int old_N = N;
    N = K;      // 新 N = 原始 K，即输出列数
    K = old_N;  // 新 K = 原始 N，即归约维长度
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        (N + 128 - 1) / 128,
        (M + 128 - 1) / 128
    );
    /*
    float4 快速路径要求：
    1. A、B、C 基地址满足 16 字节对齐；
    2. A 的行跨度 K 是 4 的倍数；
    3. B、C 的行跨度 N 是 4 的倍数。
    */
    const uintptr_t combined_pointer_bits =
        reinterpret_cast<uintptr_t>(A)
        | reinterpret_cast<uintptr_t>(B)
        | reinterpret_cast<uintptr_t>(C);
    const bool can_use_float4 =
        ((combined_pointer_bits & 0xF) == 0)
        && ((K & 3) == 0)
        && ((N & 3) == 0);
    if (can_use_float4) {
        matrix_multiplication_kernel_v4<true><<<blocks_per_grid, threads_per_block>>>(A,B,C,M,N,K);
    } else {
        matrix_multiplication_kernel_v4<false><<<blocks_per_grid, threads_per_block>>>(A,B,C,M,N,K);
    }


    cudaDeviceSynchronize();
}
