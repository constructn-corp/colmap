#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_spherical_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SphericalResJacFirstKernel(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc, double *wh,
    unsigned int wh_num_alloc, double *point, unsigned int point_num_alloc,
    SharedIndex *point_indices, double *pixel, unsigned int pixel_num_alloc,
    double *out_res, unsigned int out_res_num_alloc, double *const out_rTr,
    double *out_pose_jac, unsigned int out_pose_jac_num_alloc,
    double *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
    double *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, double *out_point_jac,
    unsigned int out_point_jac_num_alloc, double *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, double *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0,
         r9 = 0, r10 = 0, r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0, r16 = 0,
         r17 = 0, r18 = 0, r19 = 0, r20 = 0, r21 = 0, r22 = 0, r23 = 0, r24 = 0,
         r25 = 0, r26 = 0, r27 = 0, r28 = 0, r29 = 0, r30 = 0, r31 = 0, r32 = 0,
         r33 = 0, r34 = 0, r35 = 0, r36 = 0, r37 = 0, r38 = 0, r39 = 0, r40 = 0,
         r41 = 0, r42 = 0, r43 = 0, r44 = 0, r45 = 0, r46 = 0, r47 = 0, r48 = 0,
         r49 = 0, r50 = 0, r51 = 0, r52 = 0, r53 = 0, r54 = 0, r55 = 0, r56 = 0,
         r57 = 0, r58 = 0, r59 = 0, r60 = 0, r61 = 0, r62 = 0, r63 = 0, r64 = 0,
         r65 = 0, r66 = 0, r67 = 0, r68 = 0, r69 = 0, r70 = 0, r71 = 0, r72 = 0,
         r73 = 0, r74 = 0, r75 = 0, r76 = 0, r77 = 0, r78 = 0, r79 = 0, r80 = 0,
         r81 = 0, r82 = 0, r83 = 0, r84 = 0, r85 = 0, r86 = 0, r87 = 0, r88 = 0,
         r89 = 0, r90 = 0, r91 = 0, r92 = 0;

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
  };
  LoadShared<2, double, double>(pose, 0 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r9, r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r11, r12);
  };
  LoadShared<2, double, double>(pose, 2 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r13, r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r15, r16);
    r17 = r13 * r15;
    r18 = -1.00000000000000000e+00;
    r17 = fma(r18, r17, r10 * r12);
    r17 = fma(r14, r16, r17);
    r17 = fma(r9, r11, r17);
    r19 = r8 * r17;
    r19 = r19 * r17;
    r20 = 1.00000000000000000e+00;
    r21 = fma(r10, r15, r13 * r12);
    r22 = r9 * r16;
    r21 = fma(r18, r22, r21);
    r21 = fma(r14, r11, r21);
    r22 = r21 * r21;
    r22 = fma(r8, r22, r20);
    r23 = r19 + r22;
    r23 = fma(r6, r23, r4);
    r24 = fma(r14, r15, r9 * r12);
    r25 = r10 * r11;
    r24 = fma(r18, r25, r24);
    r24 = fma(r13, r16, r24);
    r25 = 2.00000000000000000e+00;
    r26 = r25 * r17;
    r27 = r24 * r26;
    r28 = fma(r10, r16, r9 * r15);
    r28 = fma(r13, r11, r28);
    r28 = fma(r18, r28, r14 * r12);
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
  };
  LoadShared<1, double, double>(pose, 6 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r34);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r35 = r15 * r11;
    r35 = r35 * r25;
    r36 = r16 * r12;
    r37 = fma(r25, r36, r35);
  };
  LoadShared<2, double, double>(pose, 4 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r38, r39);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r40 = r11 * r12;
    r41 = r15 * r16;
    r41 = r41 * r25;
    r40 = fma(r8, r40, r41);
    r42 = r11 * r11;
    r42 = r8 * r42;
    r43 = r20 + r42;
    r44 = r16 * r16;
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
    r35 = r15 * r15;
    r35 = r8 * r35;
    r44 = r44 + r35;
    r32 = r16 * r11;
    r32 = r32 * r25;
    r37 = r15 * r12;
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
    r27 = r11 * r12;
    r27 = fma(r25, r27, r41);
    r42 = r20 + r42;
    r42 = r42 + r35;
    r35 = r15 * r12;
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
    r18 = fma(r18, r18, r19 * r19);
  };
  SumStore<double>(out_rTr_local, (double *)inout_shared, 0,
                   global_thread_idx < problem_size, r18);
  if (global_thread_idx < problem_size) {
    r18 = 1.59154943091895346e-01;
    r19 = 1.00000000000000008e-15;
    r23 = fma(r10, r15, r13 * r12);
    r37 = r9 * r16;
    r2 = -1.00000000000000000e+00;
    r23 = fma(r2, r37, r23);
    r23 = fma(r14, r11, r23);
    r37 = 2.00000000000000000e+00;
    r44 = r23 * r37;
    r43 = r9 * r12;
    r17 = r14 * r15;
    r22 = r43 + r17;
    r29 = r13 * r16;
    r35 = r10 * r11;
    r22 = fma(r2, r35, r22);
    r22 = r22 + r29;
    r44 = r44 * r22;
    r35 = -2.00000000000000000e+00;
    r42 = r13 * r15;
    r42 = fma(r2, r42, r10 * r12);
    r42 = fma(r14, r16, r42);
    r42 = fma(r9, r11, r42);
    r27 = r35 * r42;
    r30 = fma(r10, r16, r9 * r15);
    r30 = fma(r13, r11, r30);
    r30 = fma(r2, r30, r14 * r12);
    r27 = fma(r30, r27, r44);
    r27 = fma(r6, r27, r40);
    r26 = r15 * r11;
    r26 = r26 * r37;
    r32 = r16 * r12;
    r32 = fma(r35, r32, r26);
    r8 = r15 * r15;
    r8 = r8 * r35;
    r20 = 1.00000000000000000e+00;
    r41 = r16 * r16;
    r41 = fma(r35, r41, r20);
    r28 = r8 + r41;
    r36 = r16 * r11;
    r36 = r36 * r37;
    r45 = r15 * r12;
    r45 = fma(r37, r45, r36);
    r46 = r23 * r37;
    r46 = r46 * r42;
    r47 = r37 * r30;
    r48 = fma(r22, r47, r46);
    r49 = r35 * r42;
    r49 = r49 * r42;
    r50 = r20 + r49;
    r51 = r35 * r22;
    r51 = r51 * r22;
    r50 = r50 + r51;
    r27 = fma(r38, r32, r27);
    r27 = fma(r34, r28, r27);
    r27 = fma(r39, r45, r27);
    r27 = fma(r7, r48, r27);
    r27 = fma(r31, r50, r27);
    r50 = copysign(r19, r27);
    r50 = r50 + r27;
    r48 = 1.0 / r50;
    r52 = r10 * r11;
    r53 = 5.00000000000000000e-01;
    r54 = -5.00000000000000000e-01;
    r52 = fma(r54, r43, r53 * r52);
    r52 = fma(r54, r17, r52);
    r52 = fma(r54, r29, r52);
    r55 = r42 * r52;
    r56 = r13 * r12;
    r57 = r10 * r15;
    r57 = fma(r53, r57, r53 * r56);
    r56 = r9 * r16;
    r57 = fma(r54, r56, r57);
    r58 = r14 * r11;
    r57 = fma(r53, r58, r57);
    r58 = fma(r57, r47, r37 * r55);
    r56 = r37 * r22;
    r59 = r13 * r15;
    r60 = r14 * r16;
    r60 = fma(r54, r60, r53 * r59);
    r59 = r9 * r11;
    r60 = fma(r54, r59, r60);
    r61 = r10 * r54;
    r60 = fma(r12, r61, r60);
    r59 = r23 * r37;
    r62 = r14 * r12;
    r63 = r9 * r15;
    r63 = fma(r54, r63, r53 * r62);
    r62 = r13 * r11;
    r63 = fma(r54, r62, r63);
    r63 = fma(r16, r61, r63);
    r59 = r59 * r63;
    r56 = fma(r60, r56, r59);
    r58 = r58 + r56;
    r62 = r37 * r42;
    r62 = r62 * r63;
    r64 = r37 * r22;
    r64 = r64 * r57;
    r65 = r62 + r64;
    r66 = r35 * r30;
    r65 = fma(r60, r66, r65);
    r67 = r35 * r23;
    r65 = fma(r52, r67, r65);
    r65 = fma(r7, r65, r31 * r58);
    r58 = r23 * r60;
    r66 = -4.00000000000000000e+00;
    r58 = r58 * r66;
    r68 = r42 * r57;
    r68 = r68 * r66;
    r69 = r58 + r68;
    r65 = fma(r6, r69, r65);
    r69 = r37 * r42;
    r69 = r69 * r60;
    r70 = r23 * r37;
    r70 = fma(r57, r70, r69);
    r71 = r37 * r22;
    r71 = r71 * r52;
    r72 = r63 * r47;
    r73 = r71 + r72;
    r74 = r70 + r73;
    r75 = r35 * r30;
    r75 = fma(r35, r55, r57 * r75);
    r75 = r75 + r56;
    r75 = fma(r6, r75, r7 * r74);
    r74 = r22 * r66;
    r57 = r63 * r74;
    r68 = r68 + r57;
    r75 = fma(r31, r68, r75);
    r49 = r20 + r49;
    r68 = r23 * r67;
    r49 = r49 + r68;
    r49 = fma(r6, r49, r4);
    r76 = r37 * r42;
    r76 = r76 * r22;
    r77 = fma(r30, r67, r76);
    r44 = fma(r42, r47, r44);
    r78 = r16 * r12;
    r78 = fma(r37, r78, r26);
    r26 = r11 * r12;
    r79 = r15 * r16;
    r79 = r79 * r37;
    r26 = fma(r35, r26, r79);
    r80 = r11 * r11;
    r80 = r35 * r80;
    r41 = r80 + r41;
    r49 = fma(r7, r77, r49);
    r49 = fma(r31, r44, r49);
    r49 = fma(r34, r78, r49);
    r49 = fma(r39, r26, r49);
    r49 = fma(r38, r41, r49);
    r44 = r2 * r49;
    r77 = r50 * r50;
    r81 = 1.0 / r77;
    r44 = r44 * r81;
    r81 = fma(r75, r44, r65 * r48);
    r82 = r18 * r81;
    r83 = r49 * r49;
    r84 = r77 + r83;
    r85 = 1.0 / r84;
    r86 = r0 * r77;
    r82 = r82 * r85;
    r82 = r82 * r86;
    r87 = -3.18309886183790691e-01;
    r88 = r37 * r49;
    r89 = r37 * r27;
    r89 = fma(r75, r89, r65 * r88);
    r76 = fma(r23, r47, r76);
    r76 = fma(r6, r76, r5);
    r88 = r11 * r12;
    r88 = fma(r37, r88, r79);
    r8 = r20 + r8;
    r8 = r8 + r80;
    r80 = r15 * r12;
    r80 = fma(r35, r80, r36);
    r36 = r35 * r22;
    r36 = fma(r30, r36, r46);
    r51 = r20 + r51;
    r51 = r51 + r68;
    r76 = fma(r38, r88, r76);
    r76 = fma(r39, r8, r76);
    r76 = fma(r34, r80, r76);
    r76 = fma(r31, r36, r76);
    r76 = fma(r7, r51, r76);
    r51 = r53 * r76;
    r83 = r19 + r83;
    r83 = fma(r27, r27, r83);
    r36 = sqrt(r83);
    r19 = copysign(r19, r36);
    r36 = r19 + r36;
    r19 = r36 * r36;
    r68 = 1.0 / r19;
    r83 = rsqrt(r83);
    r51 = r51 * r68;
    r51 = r51 * r83;
    r83 = r35 * r22;
    r68 = r35 * r30;
    r68 = r68 * r63;
    r83 = fma(r52, r83, r68);
    r83 = r83 + r70;
    r57 = r58 + r57;
    r57 = fma(r7, r57, r31 * r83);
    r64 = fma(r60, r47, r64);
    r83 = r23 * r37;
    r83 = fma(r52, r83, r62);
    r64 = r64 + r83;
    r57 = fma(r6, r64, r57);
    r64 = r2 * r57;
    r62 = 1.0 / r36;
    r64 = fma(r62, r64, r89 * r51);
    r89 = r87 * r64;
    r58 = fma(r76, r76, r19);
    r70 = 1.0 / r58;
    r20 = r1 * r19;
    r89 = r89 * r70;
    r89 = r89 * r20;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r82, r89);
    r89 = r35 * r42;
    r89 = fma(r60, r89, r68);
    r82 = r23 * r37;
    r46 = r13 * r12;
    r79 = r9 * r16;
    r79 = fma(r53, r79, r54 * r46);
    r46 = r14 * r11;
    r79 = fma(r54, r46, r79);
    r79 = fma(r15, r61, r79);
    r82 = r82 * r79;
    r46 = r37 * r22;
    r17 = fma(r53, r17, r53 * r43);
    r17 = fma(r53, r29, r17);
    r17 = fma(r11, r61, r17);
    r46 = fma(r17, r46, r82);
    r89 = r89 + r46;
    r61 = r37 * r42;
    r61 = r61 * r17;
    r29 = fma(r79, r47, r61);
    r29 = r29 + r56;
    r29 = fma(r7, r29, r6 * r89);
    r89 = r42 * r63;
    r89 = r89 * r66;
    r56 = r79 * r74;
    r43 = r89 + r56;
    r29 = fma(r31, r43, r29);
    r72 = r69 + r72;
    r72 = r72 + r46;
    r46 = r23 * r66;
    r46 = r46 * r17;
    r89 = r89 + r46;
    r89 = fma(r6, r89, r31 * r72);
    r72 = r35 * r30;
    r72 = fma(r60, r67, r17 * r72);
    r69 = r37 * r22;
    r69 = r69 * r63;
    r43 = r37 * r42;
    r43 = fma(r79, r43, r69);
    r72 = r72 + r43;
    r89 = fma(r7, r72, r89);
    r72 = fma(r89, r48, r29 * r44);
    r75 = r18 * r72;
    r75 = r75 * r85;
    r75 = r75 * r86;
    r61 = r59 + r61;
    r59 = r35 * r22;
    r61 = fma(r60, r59, r61);
    r65 = r35 * r30;
    r61 = fma(r79, r65, r61);
    r65 = r23 * r37;
    r17 = fma(r17, r47, r60 * r65);
    r17 = r17 + r43;
    r17 = fma(r6, r17, r31 * r61);
    r56 = r46 + r56;
    r17 = fma(r7, r56, r17);
    r56 = r2 * r17;
    r46 = r37 * r49;
    r61 = r37 * r27;
    r61 = fma(r29, r61, r89 * r46);
    r61 = fma(r61, r51, r62 * r56);
    r56 = r87 * r61;
    r56 = r56 * r70;
    r56 = r56 * r20;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r75, r56);
    r56 = r10 * r12;
    r75 = r13 * r15;
    r75 = fma(r54, r75, r53 * r56);
    r56 = r14 * r16;
    r75 = fma(r53, r56, r75);
    r54 = r9 * r11;
    r75 = fma(r53, r54, r75);
    r74 = r75 * r74;
    r55 = r66 * r55;
    r54 = r74 + r55;
    r56 = r23 * r37;
    r56 = r56 * r75;
    r69 = r69 + r56;
    r46 = r35 * r42;
    r69 = fma(r79, r46, r69);
    r29 = r35 * r30;
    r69 = fma(r52, r29, r69);
    r69 = fma(r6, r69, r31 * r54);
    r54 = r37 * r22;
    r54 = fma(r75, r47, r79 * r54);
    r54 = r54 + r83;
    r69 = fma(r7, r54, r69);
    r68 = r71 + r68;
    r71 = r37 * r42;
    r71 = r71 * r75;
    r68 = r68 + r71;
    r68 = fma(r79, r67, r68);
    r63 = r23 * r63;
    r63 = r63 * r66;
    r55 = r63 + r55;
    r55 = fma(r6, r55, r7 * r68);
    r47 = fma(r52, r47, r56);
    r47 = r47 + r43;
    r55 = fma(r31, r47, r55);
    r47 = fma(r55, r48, r69 * r44);
    r43 = r18 * r47;
    r43 = r43 * r85;
    r43 = r43 * r86;
    r52 = r37 * r27;
    r56 = r37 * r49;
    r56 = fma(r55, r56, r69 * r52);
    r71 = r82 + r71;
    r71 = r71 + r73;
    r73 = r35 * r22;
    r82 = r35 * r30;
    r82 = fma(r75, r82, r79 * r73);
    r82 = r82 + r83;
    r82 = fma(r31, r82, r6 * r71);
    r74 = r63 + r74;
    r82 = fma(r7, r74, r82);
    r74 = r2 * r82;
    r74 = fma(r62, r74, r56 * r51);
    r56 = r87 * r74;
    r56 = r56 * r70;
    r56 = r56 * r20;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r43, r56);
    r56 = fma(r41, r48, r32 * r44);
    r43 = r18 * r56;
    r43 = r43 * r85;
    r43 = r43 * r86;
    r63 = r37 * r41;
    r71 = r37 * r32;
    r71 = fma(r27, r71, r49 * r63);
    r63 = r2 * r88;
    r63 = fma(r62, r63, r71 * r51);
    r71 = r87 * r63;
    r71 = r71 * r70;
    r71 = r71 * r20;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r43, r71);
    r71 = fma(r26, r48, r45 * r44);
    r43 = r18 * r71;
    r43 = r43 * r85;
    r43 = r43 * r86;
    r83 = r37 * r26;
    r73 = r37 * r45;
    r73 = fma(r27, r73, r49 * r83);
    r83 = r2 * r8;
    r83 = fma(r62, r83, r73 * r51);
    r73 = r87 * r83;
    r73 = r73 * r70;
    r73 = r73 * r20;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r43, r73);
    r48 = fma(r78, r48, r28 * r44);
    r44 = r18 * r48;
    r44 = r44 * r85;
    r44 = r44 * r86;
    r73 = r37 * r78;
    r43 = r37 * r28;
    r43 = fma(r27, r43, r49 * r73);
    r73 = r2 * r80;
    r73 = fma(r62, r73, r43 * r51);
    r51 = r87 * r73;
    r51 = r51 * r70;
    r51 = r51 * r20;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r44, r51);
    r51 = 3.18309886183790691e-01;
    r76 = r2 * r76;
    r36 = atan2(r76, r36);
    r36 = fma(r87, r36, r53);
    r36 = fma(r33, r2, r1 * r36);
    r36 = r51 * r36;
    r36 = r36 * r70;
    r36 = r36 * r20;
    r70 = -1.59154943091895346e-01;
    r50 = atan2(r49, r50);
    r50 = fma(r18, r50, r53);
    r50 = fma(r3, r2, r0 * r50);
    r50 = r70 * r50;
    r50 = r50 * r85;
    r50 = r50 * r86;
    r85 = fma(r81, r50, r64 * r36);
    r70 = fma(r61, r36, r72 * r50);
    WriteSum2<double, double>((double *)inout_shared, r85, r70);
  };
  FlushSumShared<2, double>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r74, r36, r47 * r50);
    r85 = fma(r56, r50, r63 * r36);
    WriteSum2<double, double>((double *)inout_shared, r70, r85);
  };
  FlushSumShared<2, double>(out_pose_njtr, 2 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = fma(r71, r50, r83 * r36);
    r50 = fma(r48, r50, r73 * r36);
    WriteSum2<double, double>((double *)inout_shared, r85, r50);
  };
  FlushSumShared<2, double>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r81 * r81;
    r85 = 2.53302959105844473e-02;
    r85 = r0 * r85;
    r84 = r84 * r84;
    r84 = 1.0 / r84;
    r85 = r85 * r84;
    r85 = r85 * r86;
    r85 = r85 * r77;
    r77 = r64 * r64;
    r86 = 1.01321183642337789e-01;
    r86 = r1 * r86;
    r58 = r58 * r58;
    r58 = 1.0 / r58;
    r86 = r86 * r58;
    r86 = r86 * r20;
    r86 = r86 * r19;
    r77 = fma(r86, r77, r85 * r50);
    r50 = r61 * r61;
    r19 = r72 * r72;
    r19 = fma(r85, r19, r86 * r50);
    WriteSum2<double, double>((double *)inout_shared, r77, r19);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r74 * r86;
    r77 = r47 * r85;
    r47 = fma(r47, r77, r74 * r19);
    r74 = r63 * r63;
    r50 = r56 * r56;
    r50 = fma(r85, r50, r86 * r74);
    WriteSum2<double, double>((double *)inout_shared, r47, r50);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r71 * r71;
    r47 = r83 * r83;
    r47 = fma(r86, r47, r85 * r50);
    r50 = r48 * r48;
    r74 = r73 * r73;
    r74 = fma(r86, r74, r85 * r50);
    WriteSum2<double, double>((double *)inout_shared, r47, r74);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = r81 * r72;
    r47 = r64 * r61;
    r47 = fma(r86, r47, r85 * r74);
    r74 = fma(r64, r19, r81 * r77);
    WriteSum2<double, double>((double *)inout_shared, r47, r74);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = r81 * r56;
    r47 = r64 * r63;
    r47 = fma(r86, r47, r85 * r74);
    r74 = r64 * r83;
    r50 = r81 * r71;
    r50 = fma(r85, r50, r86 * r74);
    WriteSum2<double, double>((double *)inout_shared, r47, r50);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r81 * r48;
    r47 = r64 * r73;
    r47 = fma(r86, r47, r85 * r50);
    r50 = fma(r61, r19, r72 * r77);
    WriteSum2<double, double>((double *)inout_shared, r47, r50);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r61 * r63;
    r47 = r72 * r56;
    r47 = fma(r85, r47, r86 * r50);
    r50 = r61 * r83;
    r74 = r72 * r71;
    r74 = fma(r85, r74, r86 * r50);
    WriteSum2<double, double>((double *)inout_shared, r47, r74);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = r72 * r48;
    r47 = r61 * r73;
    r47 = fma(r86, r47, r85 * r74);
    r74 = fma(r56, r77, r63 * r19);
    WriteSum2<double, double>((double *)inout_shared, r47, r74);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fma(r83, r19, r71 * r77);
    r19 = fma(r73, r19, r48 * r77);
    WriteSum2<double, double>((double *)inout_shared, r74, r19);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r56 * r71;
    r74 = r63 * r83;
    r74 = fma(r86, r74, r85 * r19);
    r19 = r63 * r73;
    r77 = r56 * r48;
    r77 = fma(r85, r77, r86 * r19);
    WriteSum2<double, double>((double *)inout_shared, r74, r77);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = r71 * r48;
    r74 = r83 * r73;
    r74 = fma(r86, r74, r85 * r77);
    WriteSum1<double, double>((double *)inout_shared, r74);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = 1.59154943091895346e-01;
    r77 = fma(r10, r15, r13 * r12);
    r86 = r9 * r16;
    r85 = -1.00000000000000000e+00;
    r77 = fma(r85, r86, r77);
    r77 = fma(r14, r11, r77);
    r86 = 2.00000000000000000e+00;
    r19 = r77 * r86;
    r47 = fma(r14, r15, r9 * r12);
    r50 = r10 * r11;
    r47 = fma(r85, r50, r47);
    r47 = fma(r13, r16, r47);
    r19 = r19 * r47;
    r50 = r13 * r15;
    r50 = fma(r85, r50, r10 * r12);
    r50 = fma(r14, r16, r50);
    r50 = fma(r9, r11, r50);
    r20 = -2.00000000000000000e+00;
    r58 = fma(r10, r16, r9 * r15);
    r58 = fma(r13, r11, r58);
    r58 = fma(r85, r58, r14 * r12);
    r84 = r20 * r58;
    r36 = fma(r50, r84, r19);
    r70 = r20 * r50;
    r70 = r70 * r50;
    r18 = 1.00000000000000000e+00;
    r53 = r77 * r77;
    r53 = fma(r20, r53, r18);
    r51 = r70 + r53;
    r4 = fma(r6, r51, r4);
    r50 = r86 * r50;
    r87 = r47 * r50;
    r76 = fma(r77, r84, r87);
    r19 = fma(r58, r50, r19);
    r44 = r15 * r11;
    r44 = r44 * r86;
    r43 = r16 * r12;
    r62 = fma(r86, r43, r44);
    r75 = r11 * r12;
    r79 = r15 * r16;
    r79 = r79 * r86;
    r75 = fma(r20, r75, r79);
    r52 = r11 * r11;
    r52 = r20 * r52;
    r55 = r18 + r52;
    r69 = r16 * r16;
    r69 = r69 * r20;
    r55 = r55 + r69;
    r4 = fma(r7, r76, r4);
    r4 = fma(r31, r19, r4);
    r4 = fma(r34, r62, r4);
    r4 = fma(r39, r75, r4);
    r4 = fma(r38, r55, r4);
    r55 = r85 * r4;
    r75 = 1.00000000000000008e-15;
    r40 = fma(r6, r36, r40);
    r43 = fma(r20, r43, r44);
    r69 = r18 + r69;
    r44 = r15 * r15;
    r44 = r20 * r44;
    r69 = r69 + r44;
    r62 = r16 * r11;
    r62 = r62 * r86;
    r68 = r15 * r12;
    r68 = fma(r86, r68, r62);
    r66 = r86 * r47;
    r50 = r77 * r50;
    r66 = fma(r58, r66, r50);
    r70 = r18 + r70;
    r67 = r47 * r47;
    r67 = r20 * r67;
    r70 = r70 + r67;
    r40 = fma(r38, r43, r40);
    r40 = fma(r34, r69, r40);
    r40 = fma(r39, r68, r40);
    r40 = fma(r7, r66, r40);
    r40 = fma(r31, r70, r40);
    r68 = copysign(r75, r40);
    r68 = r68 + r40;
    r69 = r68 * r68;
    r43 = 1.0 / r69;
    r55 = r55 * r43;
    r43 = 1.0 / r68;
    r54 = fma(r51, r43, r36 * r55);
    r29 = r74 * r54;
    r46 = r4 * r4;
    r89 = r69 + r46;
    r65 = 1.0 / r89;
    r60 = r0 * r69;
    r29 = r29 * r65;
    r29 = r29 * r60;
    r59 = -3.18309886183790691e-01;
    r90 = r86 * r51;
    r91 = r86 * r36;
    r91 = fma(r40, r91, r4 * r90);
    r90 = 5.00000000000000000e-01;
    r92 = r77 * r86;
    r92 = fma(r58, r92, r87);
    r6 = fma(r6, r92, r5);
    r5 = r11 * r12;
    r5 = fma(r86, r5, r79);
    r52 = r18 + r52;
    r52 = r52 + r44;
    r44 = r15 * r12;
    r44 = fma(r20, r44, r62);
    r84 = fma(r47, r84, r50);
    r53 = r67 + r53;
    r6 = fma(r38, r5, r6);
    r6 = fma(r39, r52, r6);
    r6 = fma(r34, r44, r6);
    r6 = fma(r31, r84, r6);
    r6 = fma(r7, r53, r6);
    r7 = r90 * r6;
    r46 = r75 + r46;
    r46 = fma(r40, r40, r46);
    r31 = sqrt(r46);
    r75 = copysign(r75, r31);
    r31 = r75 + r31;
    r75 = r31 * r31;
    r44 = 1.0 / r75;
    r46 = rsqrt(r46);
    r7 = r7 * r44;
    r7 = r7 * r46;
    r46 = r85 * r92;
    r44 = 1.0 / r31;
    r46 = fma(r44, r46, r91 * r7);
    r91 = r59 * r46;
    r34 = fma(r6, r6, r75);
    r52 = 1.0 / r34;
    r39 = r1 * r75;
    r91 = r91 * r52;
    r91 = r91 * r39;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx, r29, r91);
    r91 = fma(r76, r43, r66 * r55);
    r29 = r74 * r91;
    r29 = r29 * r65;
    r29 = r29 * r60;
    r5 = r86 * r66;
    r38 = r86 * r76;
    r38 = fma(r4, r38, r40 * r5);
    r5 = r85 * r53;
    r5 = fma(r44, r5, r38 * r7);
    r38 = r59 * r5;
    r38 = r38 * r52;
    r38 = r38 * r39;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx, r29, r38);
    r55 = fma(r70, r55, r19 * r43);
    r43 = r74 * r55;
    r43 = r43 * r65;
    r43 = r43 * r60;
    r38 = r85 * r84;
    r29 = r86 * r19;
    r67 = r86 * r70;
    r67 = fma(r40, r67, r4 * r29);
    r7 = fma(r67, r7, r44 * r38);
    r67 = r59 * r7;
    r67 = r67 * r52;
    r67 = r67 * r39;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx, r43, r67);
    r67 = 3.18309886183790691e-01;
    r6 = r85 * r6;
    r31 = atan2(r6, r31);
    r31 = fma(r59, r31, r90);
    r33 = fma(r33, r85, r1 * r31);
    r33 = r67 * r33;
    r33 = r33 * r52;
    r33 = r33 * r39;
    r52 = -1.59154943091895346e-01;
    r68 = atan2(r4, r68);
    r68 = fma(r74, r68, r90);
    r3 = fma(r3, r85, r0 * r68);
    r3 = r52 * r3;
    r3 = r3 * r65;
    r3 = r3 * r60;
    r65 = fma(r54, r3, r46 * r33);
    r52 = fma(r91, r3, r5 * r33);
    WriteSum2<double, double>((double *)inout_shared, r65, r52);
  };
  FlushSumShared<2, double>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r7, r33, r55 * r3);
    WriteSum1<double, double>((double *)inout_shared, r33);
  };
  FlushSumShared<1, double>(out_point_njtr, 2 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = 2.53302959105844473e-02;
    r33 = r0 * r33;
    r89 = r89 * r89;
    r89 = 1.0 / r89;
    r33 = r33 * r89;
    r33 = r33 * r60;
    r33 = r33 * r69;
    r69 = r54 * r33;
    r60 = r46 * r46;
    r89 = 1.01321183642337789e-01;
    r89 = r1 * r89;
    r34 = r34 * r34;
    r34 = 1.0 / r34;
    r89 = r89 * r34;
    r89 = r89 * r39;
    r89 = r89 * r75;
    r60 = fma(r89, r60, r54 * r69);
    r54 = r91 * r91;
    r75 = r5 * r89;
    r5 = fma(r5, r75, r33 * r54);
    WriteSum2<double, double>((double *)inout_shared, r60, r5);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r5 = r7 * r7;
    r60 = r55 * r55;
    r60 = fma(r33, r60, r89 * r5);
    WriteSum1<double, double>((double *)inout_shared, r60);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fma(r91, r69, r46 * r75);
    r5 = r46 * r7;
    r69 = fma(r55, r69, r89 * r5);
    WriteSum2<double, double>((double *)inout_shared, r60, r69);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = r91 * r55;
    r69 = fma(r33, r69, r7 * r75);
    WriteSum1<double, double>((double *)inout_shared, r69);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SphericalResJacFirst(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc, double *wh,
    unsigned int wh_num_alloc, double *point, unsigned int point_num_alloc,
    SharedIndex *point_indices, double *pixel, unsigned int pixel_num_alloc,
    double *out_res, unsigned int out_res_num_alloc, double *const out_rTr,
    double *out_pose_jac, unsigned int out_pose_jac_num_alloc,
    double *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
    double *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, double *out_point_jac,
    unsigned int out_point_jac_num_alloc, double *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, double *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SphericalResJacFirstKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, wh, wh_num_alloc, point, point_num_alloc,
      point_indices, pixel, pixel_num_alloc, out_res, out_res_num_alloc,
      out_rTr, out_pose_jac, out_pose_jac_num_alloc, out_pose_njtr,
      out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, out_point_jac, out_point_jac_num_alloc,
      out_point_njtr, out_point_njtr_num_alloc, out_point_precond_diag,
      out_point_precond_diag_num_alloc, out_point_precond_tril,
      out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar