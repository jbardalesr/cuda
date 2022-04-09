/*
Example 3.2.3

- Declare a global function calls add
- Inside main function
    - Declare a pointer to device calls dev_c
    - Allocate memory in device with cudaMalloc()
    - Pass the parameters to add function
    - Copy result into device with cudaMemCpy() and type cudaMemcpyDeviceToHost
    - Print result
    - Free Memory cudaFree()
*/

#include <stdio.h>
#include "../common/book.h"

__global__ void add(int a, int b, int *c)
{
    *c = a * b;
}

int main(void)
{
    int c;
    // device pointer
    int *dev_c;
    // cudaMalloc: allocate memory on the device
    HANDLE_ERROR(cudaMalloc((void **)&dev_c, sizeof(int)));
    // pass the parameters
    add<<<1, 1>>>(2, 7, dev_c);
    // cudaMemcpy: copies data between host and device.
    HANDLE_ERROR(cudaMemcpy(&c,
                            dev_c,
                            sizeof(int),
                            cudaMemcpyDeviceToHost));
    printf("2 * 7 = %d\n", c);

    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    printf("Number of devices = %d\n", count);

    // cudaFree: frees memory on the device.
    cudaFree(&dev_c);
    return 0;
}
