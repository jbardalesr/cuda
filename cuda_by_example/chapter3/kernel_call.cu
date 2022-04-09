#include <iostream>

// __global__ alert the compiler that a function should be compiled to run on a device instead of the host 
__global__ void kernel(void)
{
}

int main(void)
{
    // the angle brackets denote arguments we plan to pass to the runtime system
    kernel<<<1, 1>>>();
    printf("Hello, World!\n");
    return 0;
}