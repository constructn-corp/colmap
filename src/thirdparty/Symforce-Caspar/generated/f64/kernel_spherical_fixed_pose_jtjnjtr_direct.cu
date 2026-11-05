#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_spherical_fixed_pose_jtjnjtr_direct.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SphericalFixedPoseJtjnjtrDirectKernel(
        double *point_njtr, unsigned int point_njtr_num_alloc,
        SharedIndex *point_njtr_indices, double *point_jac,
        unsigned int point_jac_num_alloc, double *const out_point_njtr,
        unsigned int out_point_njtr_num_alloc, size_t problem_size) {}

void SphericalFixedPoseJtjnjtrDirect(
    double *point_njtr, unsigned int point_njtr_num_alloc,
    SharedIndex *point_njtr_indices, double *point_jac,
    unsigned int point_jac_num_alloc, double *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SphericalFixedPoseJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      point_njtr, point_njtr_num_alloc, point_njtr_indices, point_jac,
      point_jac_num_alloc, out_point_njtr, out_point_njtr_num_alloc,
      problem_size);
}

} // namespace caspar