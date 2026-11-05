#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_spherical_fixed_pose_res_jac.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SphericalFixedPoseResJacKernel(
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc, double *wh,
    unsigned int wh_num_alloc, double *point, unsigned int point_num_alloc,
    SharedIndex *point_indices, double *pixel, unsigned int pixel_num_alloc,
    double *pose, unsigned int pose_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, double *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0,
         r9 = 0, r10 = 0, r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0, r16 = 0,
         r17 = 0, r18 = 0, r19 = 0, r20 = 0, r21 = 0, r22 = 0, r23 = 0, r24 = 0,
         r25 = 0, r26 = 0, r27 = 0, r28 = 0, r29 = 0, r30 = 0, r31 = 0, r32 = 0,
         r33 = 0, r34 = 0, r35 = 0, r36 = 0, r37 = 0, r38 = 0, r39 = 0, r40 = 0,
         r41 = 0, r42 = 0, r43 = 0, r44 = 0, r45 = 0, r46 = 0, r47 = 0, r48 = 0,
         r49 = 0, r50 = 0, r51 = 0, r52 = 0, r53 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(wh, 0 * wh_num_alloc,
                                            global_thread_idx, r0, r1);
    r2 = 5.00000000000000000e-01;
    r3 = 1.59154943091895346e-01;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r4, r5);
  };
  LoadShared<2, double, double>(point, 0 * point_num_alloc, point_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        point_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = -2.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r9, r10);
    ReadIdx2<1024, double, double, double2>(pose, 0 * pose_num_alloc,
                                            global_thread_idx, r11, r12);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r13, r14);
    ReadIdx2<1024, double, double, double2>(pose, 2 * pose_num_alloc,
                                            global_thread_idx, r15, r16);
    r17 = r13 * r15;
    r18 = -1.00000000000000000e+00;
    r17 = fma(r18, r17, r10 * r12);
    r17 = fma(r14, r16, r17);
    r17 = fma(r9, r11, r17);
    r19 = r8 * r17;
    r19 = r19 * r17;
    r20 = 1.00000000000000000e+00;
    r21 = fma(r13, r12, r10 * r15);
    r22 = r14 * r11;
    r21 = fma(r18, r22, r21);
    r21 = fma(r9, r16, r21);
    r22 = r21 * r21;
    r22 = fma(r8, r22, r20);
    r23 = r19 + r22;
    r23 = fma(r6, r23, r4);
    r24 = fma(r13, r16, r10 * r11);
    r25 = r9 * r12;
    r24 = fma(r18, r25, r24);
    r24 = fma(r14, r15, r24);
    r25 = 2.00000000000000000e+00;
    r26 = r25 * r17;
    r27 = r24 * r26;
    r28 = fma(r14, r12, r13 * r11);
    r28 = fma(r9, r15, r28);
    r28 = fma(r18, r28, r10 * r16);
    r29 = r8 * r28;
    r30 = fma(r21, r29, r27);
  };
  LoadShared<1, double, double>(point, 2 * point_num_alloc, point_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double *)inout_shared,
                        point_indices_loc[threadIdx.x].target, r31);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r32 = r21 * r25;
    r32 = r32 * r24;
    r33 = fma(r28, r26, r32);
    ReadIdx1<1024, double, double, double>(pose, 6 * pose_num_alloc,
                                           global_thread_idx, r34);
    r35 = r13 * r9;
    r35 = r35 * r25;
    r36 = r14 * r10;
    r37 = fma(r25, r36, r35);
    ReadIdx2<1024, double, double, double2>(pose, 4 * pose_num_alloc,
                                            global_thread_idx, r38, r39);
    r40 = r9 * r10;
    r41 = r13 * r14;
    r41 = r41 * r25;
    r40 = fma(r8, r40, r41);
    r42 = r9 * r9;
    r42 = r8 * r42;
    r43 = r20 + r42;
    r44 = r14 * r14;
    r44 = r44 * r8;
    r43 = r43 + r44;
    r23 = fma(r7, r30, r23);
    r23 = fma(r31, r33, r23);
    r23 = fma(r34, r37, r23);
    r23 = fma(r39, r40, r23);
    r23 = fma(r38, r43, r23);
    r43 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r40);
    r17 = fma(r17, r29, r32);
    r17 = fma(r6, r17, r40);
    r36 = fma(r8, r36, r35);
    r44 = r20 + r44;
    r35 = r13 * r13;
    r35 = r8 * r35;
    r44 = r44 + r35;
    r32 = r14 * r9;
    r32 = r32 * r25;
    r37 = r13 * r10;
    r37 = fma(r25, r37, r32);
    r33 = r25 * r24;
    r26 = r21 * r26;
    r33 = fma(r28, r33, r26);
    r19 = r20 + r19;
    r30 = r24 * r24;
    r30 = r8 * r30;
    r19 = r19 + r30;
    r17 = fma(r38, r36, r17);
    r17 = fma(r34, r44, r17);
    r17 = fma(r39, r37, r17);
    r17 = fma(r7, r33, r17);
    r17 = fma(r31, r19, r17);
    r19 = copysign(r43, r17);
    r19 = r19 + r17;
    r19 = atan2(r23, r19);
    r19 = fma(r3, r19, r2);
    ReadIdx2<1024, double, double, double2>(pixel, 0 * pixel_num_alloc,
                                            global_thread_idx, r3, r33);
    r19 = fma(r3, r18, r0 * r19);
    r37 = -3.18309886183790691e-01;
    r44 = r21 * r25;
    r44 = fma(r28, r44, r27);
    r44 = fma(r6, r44, r5);
    r27 = r9 * r10;
    r27 = fma(r25, r27, r41);
    r42 = r20 + r42;
    r42 = r42 + r35;
    r35 = r13 * r10;
    r35 = fma(r8, r35, r32);
    r29 = fma(r24, r29, r26);
    r22 = r30 + r22;
    r44 = fma(r38, r27, r44);
    r44 = fma(r39, r42, r44);
    r44 = fma(r34, r35, r44);
    r44 = fma(r31, r29, r44);
    r44 = fma(r7, r22, r44);
    r44 = r18 * r44;
    r23 = fma(r23, r23, r43);
    r23 = fma(r17, r17, r23);
    r23 = sqrt(r23);
    r43 = copysign(r43, r23);
    r23 = r43 + r23;
    r23 = atan2(r44, r23);
    r23 = fma(r37, r23, r2);
    r18 = fma(r33, r18, r1 * r23);
    WriteIdx2<1024, double, double, double2>(out_res, 0 * out_res_num_alloc,
                                             global_thread_idx, r19, r18);
    r18 = 3.18309886183790691e-01;
    r19 = 5.00000000000000000e-01;
    r23 = -3.18309886183790691e-01;
    r37 = -1.00000000000000000e+00;
    r2 = fma(r13, r16, r10 * r11);
    r44 = r9 * r12;
    r2 = fma(r37, r44, r2);
    r2 = fma(r14, r15, r2);
    r44 = 2.00000000000000000e+00;
    r43 = r13 * r15;
    r43 = fma(r37, r43, r10 * r12);
    r43 = fma(r14, r16, r43);
    r43 = fma(r9, r11, r43);
    r17 = r44 * r43;
    r22 = r2 * r17;
    r29 = fma(r13, r12, r10 * r15);
    r35 = r14 * r11;
    r29 = fma(r37, r35, r29);
    r29 = fma(r9, r16, r29);
    r35 = fma(r14, r12, r13 * r11);
    r35 = fma(r9, r15, r35);
    r35 = fma(r37, r35, r10 * r16);
    r16 = r29 * r35;
    r42 = fma(r44, r16, r22);
    r5 = fma(r6, r42, r5);
    r27 = r9 * r10;
    r30 = r13 * r14;
    r30 = r30 * r44;
    r27 = fma(r44, r27, r30);
    r26 = 1.00000000000000000e+00;
    r32 = -2.00000000000000000e+00;
    r8 = r13 * r13;
    r8 = r32 * r8;
    r20 = r26 + r8;
    r41 = r9 * r9;
    r41 = r32 * r41;
    r20 = r20 + r41;
    r28 = r14 * r9;
    r28 = r28 * r44;
    r36 = r10 * r32;
    r45 = fma(r13, r36, r28);
    r46 = r2 * r35;
    r47 = r29 * r17;
    r46 = fma(r32, r46, r47);
    r48 = r2 * r2;
    r48 = r32 * r48;
    r49 = r29 * r29;
    r49 = fma(r32, r49, r26);
    r50 = r48 + r49;
    r5 = fma(r38, r27, r5);
    r5 = fma(r39, r20, r5);
    r5 = fma(r34, r45, r5);
    r5 = fma(r31, r46, r5);
    r5 = fma(r7, r50, r5);
    r45 = r37 * r5;
    r20 = 1.00000000000000008e-15;
    r27 = r43 * r43;
    r27 = r27 * r32;
    r49 = r27 + r49;
    r4 = fma(r6, r49, r4);
    r16 = fma(r32, r16, r22);
    r22 = r44 * r2;
    r22 = r22 * r29;
    r17 = fma(r35, r17, r22);
    r51 = r13 * r9;
    r51 = r51 * r44;
    r52 = r14 * r10;
    r52 = fma(r44, r52, r51);
    r30 = fma(r9, r36, r30);
    r41 = r26 + r41;
    r53 = r14 * r14;
    r53 = r32 * r53;
    r41 = r41 + r53;
    r4 = fma(r7, r16, r4);
    r4 = fma(r31, r17, r4);
    r4 = fma(r34, r52, r4);
    r4 = fma(r39, r30, r4);
    r4 = fma(r38, r41, r4);
    r41 = r4 * r4;
    r30 = r20 + r41;
    r52 = r43 * r35;
    r52 = fma(r32, r52, r22);
    r6 = fma(r6, r52, r40);
    r36 = fma(r14, r36, r51);
    r8 = r26 + r8;
    r8 = r8 + r53;
    r53 = r13 * r10;
    r53 = fma(r44, r53, r28);
    r28 = r44 * r2;
    r28 = fma(r35, r28, r47);
    r48 = r26 + r48;
    r48 = r48 + r27;
    r6 = fma(r38, r36, r6);
    r6 = fma(r34, r8, r6);
    r6 = fma(r39, r53, r6);
    r6 = fma(r7, r28, r6);
    r6 = fma(r31, r48, r6);
    r30 = fma(r6, r6, r30);
    r31 = sqrt(r30);
    r7 = copysign(r20, r31);
    r31 = r7 + r31;
    r45 = atan2(r45, r31);
    r45 = fma(r23, r45, r19);
    r33 = fma(r33, r37, r1 * r45);
    r33 = r18 * r33;
    r18 = r31 * r31;
    r45 = fma(r5, r5, r18);
    r23 = 1.0 / r45;
    r7 = r1 * r18;
    r33 = r33 * r23;
    r33 = r33 * r7;
    r23 = r44 * r49;
    r53 = r44 * r52;
    r53 = fma(r6, r53, r4 * r23);
    r5 = r19 * r5;
    r23 = 1.0 / r18;
    r30 = rsqrt(r30);
    r5 = r5 * r23;
    r5 = r5 * r30;
    r30 = r37 * r42;
    r31 = 1.0 / r31;
    r30 = fma(r31, r30, r53 * r5);
    r53 = -1.59154943091895346e-01;
    r23 = 1.59154943091895346e-01;
    r20 = copysign(r20, r6);
    r20 = r20 + r6;
    r39 = atan2(r4, r20);
    r39 = fma(r23, r39, r19);
    r3 = fma(r3, r37, r0 * r39);
    r3 = r53 * r3;
    r53 = r20 * r20;
    r41 = r41 + r53;
    r39 = 1.0 / r41;
    r23 = r0 * r53;
    r3 = r3 * r39;
    r3 = r3 * r23;
    r39 = r37 * r4;
    r19 = 1.0 / r53;
    r39 = r39 * r19;
    r20 = 1.0 / r20;
    r19 = fma(r49, r20, r52 * r39);
    r8 = fma(r19, r3, r30 * r33);
    r34 = r44 * r28;
    r36 = r44 * r16;
    r36 = fma(r4, r36, r6 * r34);
    r34 = r37 * r50;
    r34 = fma(r31, r34, r36 * r5);
    r36 = fma(r16, r20, r28 * r39);
    r38 = fma(r36, r3, r34 * r33);
    WriteSum2<double, double>((double *)inout_shared, r8, r38);
  };
  FlushSumShared<2, double>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r48, r39, r17 * r20);
    r20 = r37 * r46;
    r38 = r44 * r17;
    r8 = r44 * r48;
    r8 = fma(r6, r8, r4 * r38);
    r5 = fma(r8, r5, r31 * r20);
    r33 = fma(r5, r33, r39 * r3);
    WriteSum1<double, double>((double *)inout_shared, r33);
  };
  FlushSumShared<1, double>(out_point_njtr, 2 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = 2.53302959105844473e-02;
    r33 = r0 * r33;
    r41 = r41 * r41;
    r41 = 1.0 / r41;
    r33 = r33 * r41;
    r33 = r33 * r23;
    r33 = r33 * r53;
    r53 = r19 * r33;
    r23 = 1.01321183642337789e-01;
    r23 = r1 * r23;
    r45 = r45 * r45;
    r45 = 1.0 / r45;
    r23 = r23 * r45;
    r23 = r23 * r7;
    r23 = r23 * r18;
    r18 = r30 * r23;
    r30 = fma(r30, r18, r19 * r53);
    r19 = r36 * r36;
    r7 = r34 * r34;
    r7 = fma(r23, r7, r33 * r19);
    WriteSum2<double, double>((double *)inout_shared, r30, r7);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r5 * r5;
    r30 = r39 * r39;
    r30 = fma(r33, r30, r23 * r7);
    WriteSum1<double, double>((double *)inout_shared, r30);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r36, r53, r34 * r18);
    r53 = fma(r39, r53, r5 * r18);
    WriteSum2<double, double>((double *)inout_shared, r30, r53);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = r34 * r5;
    r30 = r36 * r39;
    r30 = fma(r33, r30, r23 * r53);
    WriteSum1<double, double>((double *)inout_shared, r30);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
}

void SphericalFixedPoseResJac(
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc, double *wh,
    unsigned int wh_num_alloc, double *point, unsigned int point_num_alloc,
    SharedIndex *point_indices, double *pixel, unsigned int pixel_num_alloc,
    double *pose, unsigned int pose_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, double *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SphericalFixedPoseResJacKernel<<<n_blocks, 1024>>>(
      sensor_from_rig, sensor_from_rig_num_alloc, wh, wh_num_alloc, point,
      point_num_alloc, point_indices, pixel, pixel_num_alloc, pose,
      pose_num_alloc, out_res, out_res_num_alloc, out_point_njtr,
      out_point_njtr_num_alloc, out_point_precond_diag,
      out_point_precond_diag_num_alloc, out_point_precond_tril,
      out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar