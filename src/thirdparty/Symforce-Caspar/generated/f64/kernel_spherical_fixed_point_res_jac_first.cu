#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_spherical_fixed_point_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SphericalFixedPointResJacFirstKernel(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc, double *wh,
    unsigned int wh_num_alloc, double *pixel, unsigned int pixel_num_alloc,
    double *point, unsigned int point_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *const out_rTr,
    double *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
    double *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
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
         r73 = 0, r74 = 0, r75 = 0, r76 = 0, r77 = 0, r78 = 0, r79 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(wh, 0 * wh_num_alloc,
                                            global_thread_idx, r0, r1);
    r2 = 5.00000000000000000e-01;
    r3 = 1.59154943091895346e-01;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r4, r5);
    ReadIdx2<1024, double, double, double2>(point, 0 * point_num_alloc,
                                            global_thread_idx, r6, r7);
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
    ReadIdx1<1024, double, double, double>(point, 2 * point_num_alloc,
                                           global_thread_idx, r31);
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
    r18 = 3.18309886183790691e-01;
    r19 = 5.00000000000000000e-01;
    r23 = -3.18309886183790691e-01;
    r37 = -1.00000000000000000e+00;
    r2 = 2.00000000000000000e+00;
    r44 = r14 * r16;
    r43 = fma(r10, r12, r44);
    r17 = r9 * r11;
    r22 = r13 * r15;
    r43 = r43 + r17;
    r43 = fma(r37, r22, r43);
    r29 = r2 * r43;
    r35 = fma(r14, r15, r9 * r12);
    r42 = r10 * r11;
    r35 = fma(r37, r42, r35);
    r35 = fma(r13, r16, r35);
    r29 = r29 * r35;
    r42 = fma(r10, r15, r13 * r12);
    r27 = r9 * r16;
    r42 = fma(r37, r27, r42);
    r42 = fma(r14, r11, r42);
    r27 = fma(r10, r16, r9 * r15);
    r27 = fma(r13, r11, r27);
    r27 = fma(r37, r27, r14 * r12);
    r30 = r2 * r27;
    r26 = fma(r42, r30, r29);
    r26 = fma(r6, r26, r5);
    r5 = r15 * r16;
    r5 = r5 * r2;
    r32 = r11 * r12;
    r32 = fma(r2, r32, r5);
    r8 = 1.00000000000000000e+00;
    r20 = r11 * r11;
    r41 = -2.00000000000000000e+00;
    r20 = r20 * r41;
    r28 = r8 + r20;
    r36 = r15 * r15;
    r36 = r36 * r41;
    r28 = r28 + r36;
    r45 = r16 * r11;
    r45 = r45 * r2;
    r46 = r15 * r12;
    r46 = fma(r41, r46, r45);
    r47 = r42 * r2;
    r47 = r47 * r43;
    r48 = r41 * r35;
    r49 = fma(r27, r48, r47);
    r50 = r35 * r48;
    r51 = r41 * r42;
    r51 = fma(r42, r51, r8);
    r52 = r50 + r51;
    r26 = fma(r38, r32, r26);
    r26 = fma(r39, r28, r26);
    r26 = fma(r34, r46, r26);
    r26 = fma(r31, r49, r26);
    r26 = fma(r7, r52, r26);
    r52 = r37 * r26;
    r49 = 1.00000000000000008e-15;
    r53 = r41 * r43;
    r53 = r53 * r43;
    r51 = r53 + r51;
    r51 = fma(r6, r51, r4);
    r4 = r41 * r42;
    r4 = fma(r27, r4, r29);
    r29 = r42 * r2;
    r29 = r29 * r35;
    r54 = fma(r43, r30, r29);
    r55 = r15 * r11;
    r55 = r55 * r2;
    r56 = r16 * r12;
    r56 = fma(r2, r56, r55);
    r57 = r11 * r12;
    r57 = fma(r41, r57, r5);
    r20 = r8 + r20;
    r5 = r16 * r16;
    r5 = r5 * r41;
    r20 = r20 + r5;
    r51 = fma(r7, r4, r51);
    r51 = fma(r31, r54, r51);
    r51 = fma(r34, r56, r51);
    r51 = fma(r39, r57, r51);
    r51 = fma(r38, r20, r51);
    r54 = r51 * r51;
    r4 = r49 + r54;
    r58 = r41 * r43;
    r58 = fma(r27, r58, r29);
    r58 = fma(r6, r58, r40);
    r40 = r16 * r12;
    r40 = fma(r41, r40, r55);
    r36 = r8 + r36;
    r36 = r36 + r5;
    r5 = r15 * r12;
    r5 = fma(r2, r5, r45);
    r47 = fma(r35, r30, r47);
    r50 = r8 + r50;
    r50 = r50 + r53;
    r58 = fma(r38, r40, r58);
    r58 = fma(r34, r36, r58);
    r58 = fma(r39, r5, r58);
    r58 = fma(r7, r47, r58);
    r58 = fma(r31, r50, r58);
    r4 = fma(r58, r58, r4);
    r50 = sqrt(r4);
    r47 = copysign(r49, r50);
    r50 = r47 + r50;
    r52 = atan2(r52, r50);
    r52 = fma(r23, r52, r19);
    r33 = fma(r33, r37, r1 * r52);
    r33 = r18 * r33;
    r18 = r50 * r50;
    r52 = fma(r26, r26, r18);
    r23 = 1.0 / r52;
    r47 = r1 * r18;
    r33 = r33 * r23;
    r33 = r33 * r47;
    r23 = r2 * r51;
    r39 = r9 * r12;
    r34 = -5.00000000000000000e-01;
    r38 = r14 * r15;
    r38 = fma(r34, r38, r34 * r39);
    r39 = r13 * r16;
    r38 = fma(r34, r39, r38);
    r53 = r10 * r11;
    r38 = fma(r19, r53, r38);
    r53 = r43 * r38;
    r39 = r13 * r12;
    r8 = r10 * r15;
    r8 = fma(r19, r8, r19 * r39);
    r39 = r9 * r16;
    r8 = fma(r34, r39, r8);
    r45 = r14 * r11;
    r8 = fma(r19, r45, r8);
    r45 = fma(r8, r30, r2 * r53);
    r39 = r2 * r35;
    r55 = r10 * r34;
    r29 = fma(r19, r22, r12 * r55);
    r29 = fma(r34, r44, r29);
    r29 = fma(r34, r17, r29);
    r59 = r42 * r2;
    r60 = r14 * r12;
    r61 = r9 * r15;
    r61 = fma(r34, r61, r19 * r60);
    r60 = r13 * r11;
    r61 = fma(r34, r60, r61);
    r61 = fma(r16, r55, r61);
    r59 = r59 * r61;
    r39 = fma(r29, r39, r59);
    r45 = r45 + r39;
    r60 = r2 * r43;
    r60 = r60 * r61;
    r62 = r2 * r35;
    r62 = r62 * r8;
    r63 = r60 + r62;
    r64 = r41 * r42;
    r63 = fma(r38, r64, r63);
    r65 = r41 * r27;
    r63 = fma(r29, r65, r63);
    r63 = fma(r7, r63, r31 * r45);
    r45 = r43 * r8;
    r65 = -4.00000000000000000e+00;
    r45 = r45 * r65;
    r64 = r42 * r29;
    r66 = r65 * r64;
    r67 = r45 + r66;
    r63 = fma(r6, r67, r63);
    r67 = r2 * r58;
    r68 = r2 * r43;
    r68 = r68 * r29;
    r69 = r42 * r2;
    r69 = fma(r8, r69, r68);
    r70 = r2 * r35;
    r70 = r70 * r38;
    r71 = r61 * r30;
    r72 = r70 + r71;
    r73 = r69 + r72;
    r74 = r41 * r27;
    r74 = fma(r41, r53, r8 * r74);
    r74 = r74 + r39;
    r74 = fma(r6, r74, r7 * r73);
    r73 = r61 * r65;
    r8 = r35 * r73;
    r45 = r45 + r8;
    r74 = fma(r31, r45, r74);
    r67 = fma(r74, r67, r63 * r23);
    r26 = r19 * r26;
    r23 = 1.0 / r18;
    r4 = rsqrt(r4);
    r26 = r26 * r23;
    r26 = r26 * r4;
    r4 = r41 * r27;
    r4 = r4 * r61;
    r23 = fma(r38, r48, r4);
    r23 = r23 + r69;
    r8 = r66 + r8;
    r8 = fma(r7, r8, r31 * r23);
    r62 = fma(r29, r30, r62);
    r23 = r42 * r2;
    r23 = fma(r38, r23, r60);
    r62 = r62 + r23;
    r8 = fma(r6, r62, r8);
    r62 = r37 * r8;
    r50 = 1.0 / r50;
    r62 = fma(r50, r62, r67 * r26);
    r67 = -1.59154943091895346e-01;
    r60 = 1.59154943091895346e-01;
    r49 = copysign(r49, r58);
    r49 = r49 + r58;
    r66 = atan2(r51, r49);
    r66 = fma(r60, r66, r19);
    r3 = fma(r3, r37, r0 * r66);
    r3 = r67 * r3;
    r67 = r49 * r49;
    r54 = r54 + r67;
    r66 = 1.0 / r54;
    r60 = r0 * r67;
    r3 = r3 * r66;
    r3 = r3 * r60;
    r49 = 1.0 / r49;
    r66 = r37 * r51;
    r69 = 1.0 / r67;
    r66 = r66 * r69;
    r74 = fma(r74, r66, r63 * r49);
    r63 = fma(r74, r3, r62 * r33);
    r69 = r41 * r43;
    r69 = fma(r29, r69, r4);
    r45 = r42 * r2;
    r75 = r13 * r12;
    r76 = r9 * r16;
    r76 = fma(r19, r76, r34 * r75);
    r75 = r14 * r11;
    r76 = fma(r34, r75, r76);
    r76 = fma(r15, r55, r76);
    r45 = r45 * r76;
    r75 = r2 * r35;
    r77 = r9 * r12;
    r78 = r14 * r15;
    r78 = fma(r19, r78, r19 * r77);
    r77 = r13 * r16;
    r78 = fma(r19, r77, r78);
    r78 = fma(r11, r55, r78);
    r75 = fma(r78, r75, r45);
    r69 = r69 + r75;
    r55 = r2 * r43;
    r55 = r55 * r78;
    r77 = fma(r76, r30, r55);
    r77 = r77 + r39;
    r77 = fma(r7, r77, r6 * r69);
    r69 = r35 * r65;
    r69 = r69 * r76;
    r39 = r43 * r73;
    r79 = r69 + r39;
    r77 = fma(r31, r79, r77);
    r71 = r68 + r71;
    r71 = r71 + r75;
    r75 = r42 * r65;
    r75 = r75 * r78;
    r39 = r75 + r39;
    r39 = fma(r6, r39, r31 * r71);
    r71 = r41 * r27;
    r71 = fma(r41, r64, r78 * r71);
    r68 = r2 * r35;
    r68 = r68 * r61;
    r61 = r2 * r43;
    r61 = fma(r76, r61, r68);
    r71 = r71 + r61;
    r39 = fma(r7, r71, r39);
    r71 = fma(r39, r49, r77 * r66);
    r55 = r59 + r55;
    r59 = r41 * r27;
    r55 = fma(r76, r59, r55);
    r55 = fma(r29, r48, r55);
    r78 = fma(r78, r30, r2 * r64);
    r78 = r78 + r61;
    r78 = fma(r6, r78, r31 * r55);
    r75 = r69 + r75;
    r78 = fma(r7, r75, r78);
    r75 = r37 * r78;
    r69 = r2 * r51;
    r55 = r2 * r58;
    r55 = fma(r77, r55, r39 * r69);
    r55 = fma(r55, r26, r50 * r75);
    r75 = fma(r55, r33, r71 * r3);
    WriteSum2<double, double>((double *)inout_shared, r63, r75);
  };
  FlushSumShared<2, double>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = r35 * r65;
    r63 = r10 * r12;
    r22 = fma(r34, r22, r19 * r63);
    r22 = fma(r19, r44, r22);
    r22 = fma(r19, r17, r22);
    r75 = r75 * r22;
    r53 = r65 * r53;
    r65 = r75 + r53;
    r17 = r42 * r2;
    r17 = r17 * r22;
    r68 = r68 + r17;
    r19 = r41 * r43;
    r68 = fma(r76, r19, r68);
    r44 = r41 * r27;
    r68 = fma(r38, r44, r68);
    r68 = fma(r6, r68, r31 * r65);
    r65 = r2 * r35;
    r65 = fma(r22, r30, r76 * r65);
    r65 = r65 + r23;
    r68 = fma(r7, r65, r68);
    r4 = r70 + r4;
    r70 = r2 * r43;
    r70 = r70 * r22;
    r65 = r41 * r42;
    r4 = fma(r76, r65, r4);
    r4 = r4 + r70;
    r73 = r42 * r73;
    r53 = r53 + r73;
    r53 = fma(r6, r53, r7 * r4);
    r30 = fma(r38, r30, r17);
    r30 = r30 + r61;
    r53 = fma(r31, r30, r53);
    r30 = fma(r53, r49, r68 * r66);
    r61 = r2 * r58;
    r38 = r2 * r51;
    r38 = fma(r53, r38, r68 * r61);
    r70 = r45 + r70;
    r70 = r70 + r72;
    r72 = r41 * r27;
    r48 = fma(r76, r48, r22 * r72);
    r48 = r48 + r23;
    r48 = fma(r31, r48, r6 * r70);
    r73 = r75 + r73;
    r48 = fma(r7, r73, r48);
    r73 = r37 * r48;
    r73 = fma(r50, r73, r38 * r26);
    r38 = fma(r73, r33, r30 * r3);
    r7 = r2 * r20;
    r75 = r2 * r40;
    r75 = fma(r58, r75, r51 * r7);
    r7 = r37 * r32;
    r7 = fma(r50, r7, r75 * r26);
    r75 = fma(r20, r49, r40 * r66);
    r31 = fma(r75, r3, r7 * r33);
    WriteSum2<double, double>((double *)inout_shared, r38, r31);
  };
  FlushSumShared<2, double>(out_pose_njtr, 2 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r2 * r57;
    r38 = r2 * r5;
    r38 = fma(r58, r38, r51 * r31);
    r31 = r37 * r28;
    r31 = fma(r50, r31, r38 * r26);
    r38 = fma(r57, r49, r5 * r66);
    r70 = fma(r38, r3, r31 * r33);
    r6 = r2 * r56;
    r23 = r2 * r36;
    r23 = fma(r58, r23, r51 * r6);
    r6 = r37 * r46;
    r6 = fma(r50, r6, r23 * r26);
    r49 = fma(r56, r49, r36 * r66);
    r3 = fma(r49, r3, r6 * r33);
    WriteSum2<double, double>((double *)inout_shared, r70, r3);
  };
  FlushSumShared<2, double>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r74 * r74;
    r70 = 2.53302959105844473e-02;
    r70 = r0 * r70;
    r54 = r54 * r54;
    r54 = 1.0 / r54;
    r70 = r70 * r54;
    r70 = r70 * r60;
    r70 = r70 * r67;
    r67 = r62 * r62;
    r60 = 1.01321183642337789e-01;
    r60 = r1 * r60;
    r52 = r52 * r52;
    r52 = 1.0 / r52;
    r60 = r60 * r52;
    r60 = r60 * r47;
    r60 = r60 * r18;
    r67 = fma(r60, r67, r70 * r3);
    r3 = r55 * r55;
    r18 = r71 * r71;
    r18 = fma(r70, r18, r60 * r3);
    WriteSum2<double, double>((double *)inout_shared, r67, r18);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = r73 * r73;
    r67 = r30 * r30;
    r67 = fma(r70, r67, r60 * r18);
    r18 = r7 * r7;
    r3 = r75 * r75;
    r3 = fma(r70, r3, r60 * r18);
    WriteSum2<double, double>((double *)inout_shared, r67, r3);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r38 * r38;
    r67 = r31 * r31;
    r67 = fma(r60, r67, r70 * r3);
    r3 = r49 * r70;
    r18 = r6 * r60;
    r6 = fma(r6, r18, r49 * r3);
    WriteSum2<double, double>((double *)inout_shared, r67, r6);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r74 * r71;
    r67 = r62 * r55;
    r67 = fma(r60, r67, r70 * r6);
    r6 = r74 * r30;
    r49 = r62 * r73;
    r49 = fma(r60, r49, r70 * r6);
    WriteSum2<double, double>((double *)inout_shared, r67, r49);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r74 * r75;
    r67 = r62 * r7;
    r67 = fma(r60, r67, r70 * r49);
    r49 = r62 * r31;
    r6 = r74 * r38;
    r6 = fma(r70, r6, r60 * r49);
    WriteSum2<double, double>((double *)inout_shared, r67, r6);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r62, r18, r74 * r3);
    r67 = r71 * r30;
    r49 = r55 * r73;
    r49 = fma(r60, r49, r70 * r67);
    WriteSum2<double, double>((double *)inout_shared, r6, r49);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r55 * r7;
    r6 = r71 * r75;
    r6 = fma(r70, r6, r60 * r49);
    r49 = r55 * r31;
    r67 = r71 * r38;
    r67 = fma(r70, r67, r60 * r49);
    WriteSum2<double, double>((double *)inout_shared, r6, r67);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = fma(r55, r18, r71 * r3);
    r6 = r73 * r7;
    r49 = r30 * r75;
    r49 = fma(r70, r49, r60 * r6);
    WriteSum2<double, double>((double *)inout_shared, r67, r49);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r30 * r38;
    r67 = r73 * r31;
    r67 = fma(r60, r67, r70 * r49);
    r49 = fma(r73, r18, r30 * r3);
    WriteSum2<double, double>((double *)inout_shared, r67, r49);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r75 * r38;
    r67 = r7 * r31;
    r67 = fma(r60, r67, r70 * r49);
    r49 = fma(r75, r3, r7 * r18);
    WriteSum2<double, double>((double *)inout_shared, r67, r49);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = fma(r31, r18, r38 * r3);
    WriteSum1<double, double>((double *)inout_shared, r18);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SphericalFixedPointResJacFirst(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc, double *wh,
    unsigned int wh_num_alloc, double *pixel, unsigned int pixel_num_alloc,
    double *point, unsigned int point_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *const out_rTr,
    double *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
    double *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SphericalFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, wh, wh_num_alloc, pixel, pixel_num_alloc,
      point, point_num_alloc, out_res, out_res_num_alloc, out_rTr,
      out_pose_njtr, out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, problem_size);
}

} // namespace caspar