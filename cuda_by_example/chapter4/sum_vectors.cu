/*
Example 3.2.3

- Define size of array
- Inside main function
    - Allocate the memory on the GPU with cudaMalloc
    - Fill the arrays 'a' and 'b' on the CPU
    - Copy the arrays 'a' and 'b' on the GPU with cudaMemCpy() and type cudaMemcpyHostToDevice
    - Apply add function
    - Copy the array 'c' back from the GPU to the CPU
    - Display the results
    - Free the memory allocated on the GPU
*/

#include "../common/book.h"

#define N 10

__global__ void add(int *a, int *b, int *c)
{
    // blockIdx contains the value of the block  index for whichever block is currently running the device code
    int tid = blockIdx.x; // handle the data at this index
    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

int main(void)
{
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // allocate the memory on the GPU
    HANDLE_ERROR(cudaMalloc((void **)&dev_a, sizeof(int) * N));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b, sizeof(int) * N));
    HANDLE_ERROR(cudaMalloc((void **)&dev_c, sizeof(int) * N));

    // fill the arrays 'a' and 'b' on the CPU
    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    // copy the arrays 'a' and 'b' on the GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice));

    // execute the device code
    add<<<N, 1>>>(dev_a, dev_b, dev_c);

    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost));

    // display the results
    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // free the memory allocated on the GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}