// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad, James E. Gonzales II
//
// ENDLICENSE

#include "CudaVerletIntegrator.h"
#include <iostream>

CudaVerletIntegrator::CudaVerletIntegrator(double timeStep)
    : CudaIntegrator(timeStep) {
  m_OldXYZQ = nullptr;
  m_NewXYZQ = nullptr;
  m_IntegratorTypeName = "CudaVerletIntegrator";
}

extern __global__ void printKernel(int numAtoms, float4 *array);
extern __global__ void printKernel(int numAtoms, double4 *array);

__global__ void propagateKernel(float4 *coords, float4 *oldCoords,
                                float4 *newCoords, double4 *velMass,
                                double *force, int stride, double timeStep,
                                int numAtoms) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numAtoms) {
    // newCoords[i] = 2 * coords[i] - lastCoords[i] - forces[i] * dt * dt *
    // minv;
    // velocities[i] = (newCoords[i] - lastCoords[i]) / (2 * dt);
    // lastCoords[i] = coords[i];
    // coords[i] = newCoords[i];
    newCoords[tid].x = 2 * coords[tid].x - oldCoords[tid].x -
                       force[tid] * timeStep * timeStep * velMass[tid].w;
    newCoords[tid].y =
        2 * coords[tid].y - oldCoords[tid].y -
        force[tid + stride] * timeStep * timeStep * velMass[tid].w;
    newCoords[tid].z =
        2 * coords[tid].z - oldCoords[tid].z -
        force[tid + 2 * stride] * timeStep * timeStep * velMass[tid].w;

    velMass[tid].x = (newCoords[tid].x - oldCoords[tid].x) * 0.5 / timeStep;
    velMass[tid].y = (newCoords[tid].y - oldCoords[tid].y) * 0.5 / timeStep;
    velMass[tid].z = (newCoords[tid].z - oldCoords[tid].z) * 0.5 / timeStep;

    // if (std::abs(coords[tid].x - newCoords[tid].x ) > 0.1) printf("Int
    // tid:%d %.6f %.6f\n", tid, coords[tid].x, newCoords[tid].x);
  }
}
__global__ void initializeKernel(float4 *coords, float4 *oldCoords,
                                 float4 *newCoords, double4 *velMass,
                                 double *force, int stride, double timeStep,
                                 int numAtoms) {
  // force is only dV / dr, so it use (-force)
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // velMass[tid].x=0.0;
  // velMass[tid].y=0.0;
  // velMass[tid].z=0.0;
  if (tid < numAtoms) {
    oldCoords[tid].x = coords[tid].x - velMass[tid].x * timeStep -
                       0.5 * force[tid] * velMass[tid].w * timeStep * timeStep;
    oldCoords[tid].y =
        coords[tid].y - velMass[tid].y * timeStep -
        0.5 * force[tid + stride] * velMass[tid].w * timeStep * timeStep;
    oldCoords[tid].z =
        coords[tid].z - velMass[tid].z * timeStep -
        0.5 * force[tid + 2 * stride] * velMass[tid].w * timeStep * timeStep;
    oldCoords[tid].w = coords[tid].w;

    newCoords[tid].x = coords[tid].x + velMass[tid].x * timeStep -
                       0.5 * force[tid] * velMass[tid].w * timeStep * timeStep;
    newCoords[tid].y =
        coords[tid].y + velMass[tid].y * timeStep -
        0.5 * force[tid + stride] * velMass[tid].w * timeStep * timeStep;
    newCoords[tid].z =
        coords[tid].z + velMass[tid].z * timeStep -
        0.5 * force[tid + 2 * stride] * velMass[tid].w * timeStep * timeStep;
    newCoords[tid].w = coords[tid].w;
  }
}

void CudaVerletIntegrator::initialize(void) {
  int numAtoms = m_Context->getNumAtoms();
  // CudaIntegrator::initializeOldNewCoords(numAtoms);
  auto energy = m_Context->calculatePotentialEnergy(true);
  // std::cout << "Total energy = " << energy << "\n";
  auto force = m_Context->getForces();
  double *forceData = (double *)force->xyz();
  auto velocityMass = m_Context->getVelocityMass();
  DeviceVector<double4> vmDeviceArray = velocityMass.getDeviceArray();
  double4 *d_velMass = vmDeviceArray.data();
  auto xyzq = m_Context->getXYZQ();
  auto coords = xyzq->getDeviceXYZQ();
  auto stride = m_Context->getForceStride();
  int nThreads = 128;
  int nBlocks = (numAtoms - 1) / nThreads + 1;
  // printKernel<<<nBlocks, nThreads>>>(1000, coords);
  initializeKernel<<<nBlocks, nThreads>>>(
      xyzq->getDeviceXYZQ(), m_OldXYZQ->getDeviceXYZQ(),
      m_NewXYZQ->getDeviceXYZQ(), d_velMass, forceData, stride, m_TimeStep,
      numAtoms);
  cudaDeviceSynchronize();

  return;
}

__global__ void imageCenteringKernel(float4 *__restrict__ coords,
                                     float4 *__restrict__ oldCoords,
                                     float4 *__restrict__ newCoords, float boxx,
                                     float boxy, float boxz, int numAtoms) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numAtoms) {
    while (coords[tid].x > boxx / 2) {
      coords[tid].x -= boxx;
      oldCoords[tid].x -= boxx;
      newCoords[tid].x -= boxx;
    }
    while (coords[tid].y > boxy / 2) {
      coords[tid].y -= boxy;
      oldCoords[tid].y -= boxy;
      newCoords[tid].y -= boxy;
    }
    while (coords[tid].z > boxz / 2) {
      coords[tid].z -= boxz;
      oldCoords[tid].z -= boxz;
      newCoords[tid].z -= boxz;
    }

    while (coords[tid].x < -boxx / 2) {
      coords[tid].x += boxx;
      oldCoords[tid].x += boxx;
      newCoords[tid].x += boxx;
    }
    while (coords[tid].y < -boxy / 2) {
      coords[tid].y += boxy;
      oldCoords[tid].y += boxy;
      newCoords[tid].y += boxy;
    }
    while (coords[tid].z < -boxz / 2) {
      coords[tid].z += boxz;
      oldCoords[tid].z += boxz;
      newCoords[tid].z += boxz;
    }

    /*
    while(coords[tid].x < -boxx/2) coords[tid].x += boxx;

    while(coords[tid].y >  boxy/2) coords[tid].y -= boxy;
    while(coords[tid].y < -boxy/2) coords[tid].y += boxy;

    while(coords[tid].z >  boxz/2) coords[tid].z -= boxz;
    while(coords[tid].z < -boxz/2) coords[tid].z += boxz;
  */
  }
}

/*
void CudaVerletIntegrator::propagate(int numSteps) {
  auto energy = context->calculatePotentialEnergy(true);
  std::cout << energy << "\n";
  context->calculateKineticEnergy();
  int numAtoms = context->getNumAtoms();
  int nThreads = 128;
  // int nThreads = 512;
  int nBlocks = (numAtoms - 1) / nThreads + 1;
  auto xyzq = context->getXYZQ();
  auto stride = context->getForceStride();
  auto boxDimensions = context->getBoxDimensions();
  float boxx = boxDimensions[0];
  float boxy = boxDimensions[1];
  float boxz = boxDimensions[2];
  numSteps = 90;
  for (int i = 1; i < numSteps; ++i) {
    auto temp = oldXYZQ->getDeviceXYZQ();
    oldXYZQ->setDeviceXYZQ(xyzq->getDeviceXYZQ());
    xyzq->setDeviceXYZQ(newXYZQ->getDeviceXYZQ());
    newXYZQ->setDeviceXYZQ(temp);
    if (i % 20 == 0) {
      if (i % 100 == 0)
        imageCenteringKernel<<<nBlocks, nThreads>>>(
            xyzq->getDeviceXYZQ(), oldXYZQ->getDeviceXYZQ(),
            newXYZQ->getDeviceXYZQ(), boxx, boxy, boxz, numAtoms);
      cudaDeviceSynchronize();
      energy = context->calculatePotentialEnergy(true);
    } else {
      // energy = context->calculatePotentialEnergy(false);
      energy = context->calculatePotentialEnergy(true);
    }
    // std::cout << energy << "\n";
    auto force = context->getForces();
    double *forceData = (double *)force->xyz();
    auto velocityMass = context->getVelocityMass();
    DeviceVector<double4> vmDeviceArray = velocityMass.getDeviceArray();
    double4 *d_velMass = vmDeviceArray.data();

    propagateKernel<<<nBlocks, nThreads>>>(
        xyzq->getDeviceXYZQ(), oldXYZQ->getDeviceXYZQ(),
        newXYZQ->getDeviceXYZQ(), d_velMass, forceData, stride, timeStep,
        numAtoms);
    cudaDeviceSynchronize();
    context->calculateKineticEnergy();
    // double *ke = (double *)malloc(sizeof(double));
    // cudaMemcpy(ke, context->getKi , sizeof(double), cudaMemcpyDeviceToHost);
    // std::cout << i << " " << energy << " " << *ke <<" "<< energy + *ke<<
    // "\n";
    // std::cout << *ke << "\n";
  }
}
*/
