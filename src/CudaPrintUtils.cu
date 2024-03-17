
__global__ void printKernel(int numAtoms, float4 *array) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < numAtoms) {
    // printf("%.6f\n", (double)force[id]);
    printf("%d :( %.6f,%.6f,%.6f,%.6f)\n", id, array[id].x, array[id].y,
           array[id].z, array[id].w);
  }
}

__global__ void printKernel(int numAtoms, double4 *array) {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < numAtoms) {
    // printf("%.6f\n", (double)force[id]);
    printf("%d :( %.6f,%.6f,%.6f,%.6f)\n", id, array[id].x, array[id].y,
           array[id].z, array[id].w);
  }
}
