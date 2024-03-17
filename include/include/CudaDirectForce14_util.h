// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#ifndef NOCUDAC
//
// CUDA device functions for direct force calculation
//

/** @brief 1-4 exclusion and interaction calculation kernel
 */
template <typename AT, typename CT, int vdw_model, int elec_model,
          bool calc_energy, bool calc_virial, bool tex_vdwparam>
__global__ void CUDA_14_KERNEL_NAME(
#ifdef USE_TEXTURE_OBJECTS
    const cudaTextureObject_t vdwParam14TexObj,
#endif
    const int nin14list, const int nex14list, const int nin14block,
    const xx14list_t *in14list, const xx14list_t *ex14list,
#ifdef USE_BLOCK
    const int *in14TblBlockPos, const int *ex14TblBlockPos,
    const int *in14BlockToBlock, const int *ex14BlockToBlock,
    const int *in14BlockToTblPos, const int *ex14BlockToTblPos,
#endif
    const int *vdwtype, const float *vdwparam14, const float4 *xyzq,
#ifdef USE_BLOCK
#ifdef USE_TEXTURE_OBJECTS
    const cudaTextureObject_t blockParamTexObj,
#endif
    const int usePMEL, const float *blockParamEx,
#ifdef USE_BLOCK_SOFTCORE
    double *__restrict__ DSoftDFscale,
#endif
#else
    const float fscale,
#endif
    const int stride, AT *force, Virial_t *__restrict__ virial,
    double *__restrict__ energy_vdw, double *__restrict__ energy_elec,
    double *__restrict__ energy_excl) {
  // Amount of shared memory required:
  // blockDim.x*sizeof(double2)
  // #ifdef USE_BLOCK_SOFTCORE
  extern __shared__ double3 shpot[]; // Only need double3 if soft cores are in
                                     // use, but compiler doesn't seem smart
                                     // enough to see that, or else I've missed
                                     // a subtle bug
                                     // #else
                                     //   extern __shared__ double2 shpot[];
                                     // #endif

  if (blockIdx.x < nin14block) { // for USE_BLOCK, this is ALL nin14block
    double vdw_pot, elec_pot;
    // Need to pass soft_f, even if it's only used ifdef USE_BLOCK_SOFTCORE
    double soft_f;
    if (calc_energy) {
      vdw_pot = 0.0;
      elec_pot = 0.0;
#ifdef USE_BLOCK_SOFTCORE
      soft_f = 0.0;
#endif
    }

#ifdef USE_BLOCK
#ifdef USE_TEXTURE_OBJECTS
    float fscale =
        tex1Dfetch<float>(blockParamTexObj, in14BlockToBlock[blockIdx.x]);
#else
    float fscale = tex1Dfetch(blockParamTexRef, in14BlockToBlock[blockIdx.x]);
#endif
    int pos = threadIdx.x + in14BlockToTblPos[blockIdx.x];
    if (pos < in14TblBlockPos[in14BlockToBlock[blockIdx.x] + 1])
#else
    int pos = threadIdx.x + blockIdx.x * blockDim.x;
    if (pos < nin14list)
#endif
    {
#ifdef USE_BLOCK_SOFTCORE
      calc_in14_force_device<AT, CT, vdw_model, elec_model, calc_energy,
                             calc_virial, true, tex_vdwparam>
#else
      calc_in14_force_device<AT, CT, vdw_model, elec_model, calc_energy,
                             calc_virial, false, tex_vdwparam>
#endif
          (
#ifdef USE_TEXTURE_OBJECTS
              vdwParam14TexObj,
#endif
              pos, in14list, vdwtype, vdwparam14, xyzq, fscale, stride, force,
              vdw_pot, elec_pot, soft_f, virial);
    }

    if (calc_energy) {
      shpot[threadIdx.x].x = vdw_pot;
      shpot[threadIdx.x].y = elec_pot;
#ifdef USE_BLOCK_SOFTCORE
      shpot[threadIdx.x].z = soft_f;
#endif
      __syncthreads();
      for (int i = 1; i < blockDim.x; i *= 2) {
        int t = threadIdx.x + i;
        double val1 = (t < blockDim.x) ? shpot[t].x : 0.0;
        double val2 = (t < blockDim.x) ? shpot[t].y : 0.0;
#ifdef USE_BLOCK_SOFTCORE
        double val3 = (t < blockDim.x) ? shpot[t].z : 0.0;
#endif
        __syncthreads();
        shpot[threadIdx.x].x += val1;
        shpot[threadIdx.x].y += val2;
#ifdef USE_BLOCK_SOFTCORE
        shpot[threadIdx.x].z += val3;
#endif
        __syncthreads();
      }
      if (threadIdx.x == 0) {
#ifdef USE_BLOCK
        atomicAdd(&energy_vdw[in14BlockToBlock[blockIdx.x]], shpot[0].x);
        atomicAdd(&energy_elec[in14BlockToBlock[blockIdx.x]], shpot[0].y);
#ifdef USE_BLOCK_SOFTCORE
        atomicAdd(&DSoftDFscale[in14BlockToBlock[blockIdx.x]], shpot[0].z);
#endif
#else
        atomicAdd(energy_vdw, shpot[0].x);
        atomicAdd(energy_elec, shpot[0].y);
#endif
      }
    }

  } else if (elec_model == EWALD || elec_model == EWALD_LOOKUP) {
    // NOTE: Only Ewald potentials calculate 1-4 exclusions
    double excl_pot;
    if (calc_energy)
      excl_pot = 0.0;

#ifdef USE_BLOCK
    float fscale;
    if (usePMEL >= 2) {
      fscale = blockParamEx[ex14BlockToBlock[blockIdx.x - nin14block]];
    } else { // usePMEL==1 (If it's zero, this shouldn't be called at all.)
#ifdef USE_TEXTURE_OBJECTS
      fscale = tex1Dfetch<float>(blockParamTexObj,
                                 ex14BlockToBlock[blockIdx.x - nin14block]);
#else
      fscale = tex1Dfetch(blockParamTexRef,
                          ex14BlockToBlock[blockIdx.x - nin14block]);
#endif
    }
    int pos = threadIdx.x + ex14BlockToTblPos[blockIdx.x - nin14block];
    if (pos < ex14TblBlockPos[ex14BlockToBlock[blockIdx.x - nin14block] + 1])
#else
    int pos = threadIdx.x + (blockIdx.x - nin14block) * blockDim.x;
    if (pos < nex14list)
#endif
    {
      calc_ex14_force_device<AT, CT, elec_model, calc_energy, calc_virial>(
          pos, ex14list, xyzq, fscale, stride, force, excl_pot, virial);
    }

    if (calc_energy) {
      shpot[threadIdx.x].x = excl_pot;
      __syncthreads();
      for (int i = 1; i < blockDim.x; i *= 2) {
        int t = threadIdx.x + i;
        double val = (t < blockDim.x) ? shpot[t].x : 0.0;
        __syncthreads();
        shpot[threadIdx.x].x += val;
        __syncthreads();
      }
      if (threadIdx.x == 0) {
#ifdef USE_BLOCK
        atomicAdd(&energy_excl[ex14BlockToBlock[blockIdx.x - nin14block]],
                  shpot[0].x);
#else
        atomicAdd(energy_excl, shpot[0].x);
#endif
      }
    }
  }
}

#endif // NOCUDAC
