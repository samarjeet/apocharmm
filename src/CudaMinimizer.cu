// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Samarjeet Prasad
//
// ENDLICENSE

// #include "CharmmContext.h"
#include "CudaMinimizer.h"
// #include <Eigen/Core>
#include <cmath>
#include <iostream>

// #include <LBFGS.h>

CudaMinimizer::CudaMinimizer() : CudaIntegrator(0.0) {
  nsteps = 100; // default number of steps
  method = "sd";
  verboseFlag = false;
}

// void CudaMinimizer::setCharmmContext(std::shared_ptr<CharmmContext> csc) {
//   context = csc;
// }

static __global__ void updateSPKernel(int numAtoms, float4 *__restrict__ xyzq,
                                      const double4 *__restrict__ coords) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    xyzq[index].x = (float)coords[index].x;
    xyzq[index].y = (float)coords[index].y;
    xyzq[index].z = (float)coords[index].z;
  }
}

// class Energy {
// private:
//   int numAtoms;
//   int stride;
//   std::shared_ptr<CharmmContext> context;
//   std::shared_ptr<Force<double>> force;
//   std::vector<double> forceVec;
//   std::vector<float> gradVec;
//   // std::vector<float> coords;
//   std::vector<float> energies;

//   float prevEnergy;
//   int iter;

// public:
//   Energy(std::shared_ptr<CharmmContext> csc) {
//     context = csc;
//     numAtoms = context->getNumAtoms();
//     stride = context->getForceStride();

//     context->calculatePotentialEnergy();
//     force = context->getForces();

//     // coords.resize(numAtoms * 3);
//     forceVec.resize(stride * 3);
//     gradVec.resize(numAtoms * 3);
//     prevEnergy = 0.0f;
//     iter = 0;
//   }

/*
  Takes the coordinates as numAtoms*3 sized Eigen::VectorXd
  fills in grad with the gradient
  and returns the energy

  TODO : This is a very inefficient way of doing this.
  - use the double precision coords rather than xyzq float

  - An even more efficient way would be to use a GPU to perform the L-BFGS
  update.
*/
//   double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {

//     auto coordsContainer = context->getCoordinatesCharges();
//     auto xyzq = context->getXYZQ()->getDeviceXYZQ();

//     for (int i = 0; i < numAtoms; ++i) {
//       coordsContainer[i].x = x[3 * i];
//       coordsContainer[i].y = x[3 * i + 1];
//       coordsContainer[i].z = x[3 * i + 2];
//     }
//     std::cout << "x[0] : " << x[0] << " " << x[1] << " " << x[2] << "\n";

//     coordsContainer.transferToDevice();
//     auto coords = coordsContainer.getDeviceArray().data();
//     int numThreads = 256;
//     int numBlocks = (numAtoms - 1) / numThreads + 1;
//     updateSPKernel<<<numBlocks, numThreads>>>(numAtoms, xyzq, coords);
//     cudaCheck(cudaDeviceSynchronize());

//     // context->calculatePotentialEnergy();
//     // context->calculateForces();
//     // force = context->getForces();
//     // energies = context->getPotentialEnergies();
//     context->resetNeighborList();
//     auto e = context->calculatePotentialEnergy(true);
//     force = context->getForces();
//     std::cout << "[Min] Energy : " << ++iter << " " << e << " "
//               << prevEnergy - e << "\n";
//     prevEnergy = e;
//     cudaMemcpy(forceVec.data(), force->xyz(), stride * 3 * sizeof(double),
//                cudaMemcpyDeviceToHost);

//     for (int i = 0; i < numAtoms; i++) {
//       grad[3 * i] = forceVec[i];
//       grad[3 * i + 1] = forceVec[i + stride];
//       grad[3 * i + 2] = forceVec[i + 2 * stride];
//     }
//     // return (double)energies[0];
//     return (double)e;
//   }
// };

__global__ void steepestDescentKernel(int numAtoms, double stepSize, int stride,
                                      double4 *__restrict__ velMass,
                                      double4 *__restrict__ coords,
                                      const double *__restrict__ force) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numAtoms) {
    double fx = -force[index];
    double fy = -force[index + stride];
    double fz = -force[index + 2 * stride];

    coords[index].x += stepSize * fx; //* velMass[index].w;
    coords[index].y += stepSize * fy; //* velMass[index].w;
    coords[index].z += stepSize * fz; //* velMass[index].w;
  }
}

__global__ void norm(int numAtoms, int stride, const double *__restrict__ force,
                     double *__restrict__ result) {

  extern __shared__ double sdata[];

  size_t tid = threadIdx.x;
  size_t index = blockIdx.x * blockDim.x + tid;

  sdata[tid] = 0.0;

  for (int i = index; i < numAtoms; i += blockDim.x * gridDim.x) {
    double fx = force[i];
    double fy = force[i + stride];
    double fz = force[i + 2 * stride];

    sdata[tid] += fx * fx + fy * fy + fz * fz;
  }

  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(result, sdata[0]);
  }
}

void CudaMinimizer::initialize() {}

void CudaMinimizer::minimize(int numSteps) {
  // exit with messgae that it's still building
  // std::cout << "Minimize not implemented yet. Exiting.\n";
  // exit(1);

  // numSteps = 1000;
  double tol = 0.00001 / 1000000;
  bool toleranceReached = false;
  if (method == "sd") {
    // context->calculateForces(true, true, true);
    // auto pe = context->getPotentialEnergy();
    // pe.transferFromDevice();

    float prevEnergy = 0.0;
    float bestEnergy = prevEnergy;
    float bestGradientNorm = 0.0;

    double stepSize = 0.02;
    double s = 1.0;
    int converged = 0;
    double stepSizeTolerance = 0.0;
    double gradientTolerance = 0.0;

    auto xyzq = m_Context->getXYZQ()->getDeviceXYZQ();
    auto coords = m_Context->getCoordinatesCharges().getDeviceArray().data();
    auto velMass = m_Context->getVelocityMass().getDeviceArray().data();

    int numAtoms = m_Context->getNumAtoms();
    int stride = m_Context->getForceStride();
    int numThreads = 128;
    int numBlocks = (numAtoms - 1) / numThreads + 1;

    double *d_forceNorm;
    cudaMalloc(&d_forceNorm, sizeof(double));

    for (int iter = 0; iter < numSteps; ++iter) {

      verboseFlag = true;
      if (verboseFlag) {
        std::cout << "Iter : " << iter // << " stepSize : " << stepSize
                  << " ";
      }

      // cudaCheck(cudaStreamSynchronize(*integratorMemcpyStream));
      // cudaCheck(cudaDeviceSynchronize());

      if (m_UsingHolonomicConstraints) {
        copy_DtoD_async<double4>(coords, m_CoordsRef.getDeviceArray().data(),
                                 numAtoms, *m_IntegratorMemcpyStream);

        cudaCheck(cudaStreamSynchronize(*m_IntegratorMemcpyStream));
        cudaCheck(cudaDeviceSynchronize());

        m_HolonomicConstraint->handleHolonomicConstraints(
            m_CoordsRef.getDeviceArray().data());
        updateSPKernel<<<numBlocks, numThreads>>>(numAtoms, xyzq, coords);

        cudaCheck(cudaStreamSynchronize(*m_IntegratorMemcpyStream));
        cudaCheck(cudaDeviceSynchronize());
      }

      m_Context->resetNeighborList();
      //  context->calculateForces();
      m_Context->calculateForces(true, true, true);

      if (m_UsingHolonomicConstraints) {
        m_HolonomicConstraint->removeForceAlongHolonomicConstraints();
      }

      auto pe = m_Context->getPotentialEnergy();
      pe.transferFromDevice();
      float energy = pe[0];

      if (verboseFlag) {
        std::cout << "Energy : " << energy << " ";
      }

      auto force = m_Context->getForces();
      // calculate the force norm
      // TODO : calculate the force norm sqrt( <f,f>/dim)
      int numThreads = 1024;
      int numBlocks = 256;

      // set d_forceNorm to 0
      cudaMemset(d_forceNorm, 0, sizeof(double));
      norm<<<numBlocks, numThreads, numThreads * sizeof(double)>>>(
          numAtoms, stride, force->xyz(), d_forceNorm);
      cudaDeviceSynchronize();

      double h_forceNorm;
      cudaMemcpy(&h_forceNorm, d_forceNorm, sizeof(double),
                 cudaMemcpyDeviceToHost);
      // std::cout << "Force Norm : " << h_forceNorm << " ";

      double gradientNorm = std::sqrt(h_forceNorm / (3.0 * numAtoms));
      if (verboseFlag) {
        std::cout << " gradientNorm : " << gradientNorm << " ";
      }

      if (iter == 0) {
        prevEnergy = energy;
        bestEnergy = energy;
        bestGradientNorm = gradientNorm;
      }
      stepSize = 0.5 * stepSize;
      if (energy < prevEnergy) {
        stepSize = 2.4 * stepSize;
      }

      if (energy < bestEnergy) {
        bestEnergy = energy;
        // TODO : copy the best coordinates
      }

      if (iter > 0) {

        if (stepSize < stepSizeTolerance) {
          converged = 1;
        }
        if (gradientNorm < gradientTolerance) {
          converged = 2;
        }
        if (std::abs(energy - prevEnergy) < tol) {
          converged = 3;
        }

        if (stepSize < tol) {
          // std::cout << "[Minimization] Minimized till tolerance
          // criterion.\n";
          //  toleranceReached = true;
          //  break;
        }
      }

      s = stepSize / std::max(gradientNorm, 1e-10);
      if (verboseFlag) {
        std::cout << " stepSize : " << stepSize << " ";
      }

      // s = 0.0;

      // cudaCheck(cudaDeviceSynchronize());
      steepestDescentKernel<<<numBlocks, numThreads>>>(
          numAtoms, s, stride, velMass, coords, force->xyz());

      cudaCheck(cudaDeviceSynchronize());
      updateSPKernel<<<numBlocks, numThreads, 0, *m_IntegratorStream>>>(
          numAtoms, xyzq, coords);
      cudaCheck(cudaStreamSynchronize(*m_IntegratorStream));
      cudaCheck(cudaDeviceSynchronize());

      prevEnergy = energy;

      if (verboseFlag) {
        std::cout << "\n";
      }
      verboseFlag = false;
    }

    cudaFree(d_forceNorm);

  } else if (method == "abnr") {
    std::cout << "Not implemented yet.\n";

  } else if (method == "lbfgs") {
    // LBFGSpp::LBFGSParam<double> param;
    // param.epsilon = 1e-5;
    // param.max_iterations = numSteps;
    // std::cout << "Min " << numSteps << "\n";

    // LBFGSpp::LBFGSSolver<double> solver(param);
    // Energy energy(context);

    // Eigen::VectorXd coords = Eigen::VectorXd::Zero(3 *
    // context->getNumAtoms());

    // auto coordsContainer = context->getCoordinatesCharges();
    // coordsContainer.transferFromDevice();

    ////  Initializing with current coordinates

    // for (int i = 0; i < context->getNumAtoms(); i++) {
    //   coords[3 * i] = coordsContainer[i].x;
    //   coords[3 * i + 1] = coordsContainer[i].y;
    //   coords[3 * i + 2] = coordsContainer[i].z;
    // }

    // double energyValue;
    // try {
    //   int numIterations = solver.minimize(energy, coords, energyValue);
    // } catch (...) {
    //   std::cout << "LBFGS Error: " << std::endl;
    // }
    // auto coordss = context->getCoordinatesCharges().getHostArray();
    // for (int i = 0; i < context->getNumAtoms(); i++) {
    //   coordss[i].x = coords[3 * i];
    //   coordss[i].y = coords[3 * i + 1];
    //   coordss[i].z = coords[3 * i + 2];
    // }
    // context->getCoordinatesCharges().set(coordss);
    throw std::invalid_argument("LBFGS method DEPRECATED. Stopping.\n");
  }

  else {
    std::cout << "Not a valid method\n.";
    exit(1);
  }
  if (!toleranceReached) {
    std::cout << "[Minimization] Didn't reach minimization tolerance. Number "
                 "of iterations exhausted.\n";
  }
}

void CudaMinimizer::minimize() { minimize(nsteps); }

void CudaMinimizer::setMethod(std::string _method) { method = _method; }

void CudaMinimizer::setVerboseFlag(bool _flag) { verboseFlag = _flag; }
