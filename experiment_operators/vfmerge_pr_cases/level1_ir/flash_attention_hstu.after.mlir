#map = affine_map<(d0) -> (d0, 0)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.disable_auto_tile_and_bind_subblock, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @_attn_fwd_outlined_merged_merged_vf_0(%arg0: tensor<1x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64xf32>) -> (tensor<1x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant dense<5.000000e-01> : vector<1x64xf32>
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %c0_1 = arith.constant 0 : index
    %1 = vector.transfer_write %cst_0, %arg1[%c0_1] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0_3 = arith.constant 0 : index
    %2 = scf.for %arg4 = %c0_3 to %c64 step %c1 iter_args(%arg5 = %arg2) -> (tensor<64x64xf32>) {
      %extracted_slice = tensor.extract_slice %arg5[%arg4, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
      %4 = vector.transfer_write %cst_2, %extracted_slice[%c0_3, %c0_3] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
      %inserted_slice = tensor.insert_slice %4 into %arg5[%arg4, 0] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x64xf32>
      scf.yield %inserted_slice : tensor<64x64xf32>
    }
    %cst_4 = arith.constant dense<0xFF800000> : vector<64xf32>
    %c0_5 = arith.constant 0 : index
    %3 = vector.transfer_write %cst_4, %arg3[%c0_5] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    return %0, %1, %2, %3 : tensor<1x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>
  }
  func.func @_attn_fwd_scope_0_outlined_manual_vf_0(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = scf.for %arg7 = %arg3 to %arg4 step %arg5 iter_args(%arg8 = %arg6) -> (tensor<64x64xf32>)  : i32 {
      %1 = arith.index_cast %arg7 : i32 to index
      %extracted_slice = tensor.extract_slice %arg8[%1, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg0[%1, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
      %extracted_slice_1 = tensor.extract_slice %arg1[%1] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %extracted_slice_2 = tensor.extract_slice %arg2[%1] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %2 = vector.transfer_read %extracted_slice[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %3 = vector.transfer_read %extracted_slice_1[%c0], %cst {in_bounds = [true, true], permutation_map = #map} : tensor<1xf32>, vector<1x64xf32>
      %4 = vector.transfer_read %extracted_slice_0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %5 = vector.transfer_read %extracted_slice_2[%c0], %cst {in_bounds = [true, true], permutation_map = #map} : tensor<1xf32>, vector<1x64xf32>
      %6 = arith.mulf %2, %3 : vector<1x64xf32>
      %7 = arith.addf %6, %4 : vector<1x64xf32>
      %8 = arith.divf %7, %5 : vector<1x64xf32>
      %9 = vector.transfer_write %8, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
      %inserted_slice = tensor.insert_slice %9 into %arg8[%1, 0] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x64xf32>
      scf.yield %inserted_slice : tensor<64x64xf32>
    }
    return %0 : tensor<64x64xf32>
  }
  func.func @_attn_fwd_scope_1_outlined_manual_vf_0(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = scf.for %arg6 = %arg2 to %arg3 step %arg4 iter_args(%arg7 = %arg5) -> (tensor<64x64xf32>)  : i32 {
      %1 = arith.index_cast %arg6 : i32 to index
      %extracted_slice = tensor.extract_slice %arg7[%1, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg0[%1, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
      %extracted_slice_1 = tensor.extract_slice %arg1[%1] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %2 = vector.transfer_read %extracted_slice[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %3 = vector.transfer_read %extracted_slice_1[%c0], %cst {in_bounds = [true, true], permutation_map = #map} : tensor<1xf32>, vector<1x64xf32>
      %4 = vector.transfer_read %extracted_slice_0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %5 = arith.mulf %2, %3 : vector<1x64xf32>
      %6 = arith.addf %5, %4 : vector<1x64xf32>
      %7 = vector.transfer_write %6, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
      %inserted_slice = tensor.insert_slice %7 into %arg7[%1, 0] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x64xf32>
      scf.yield %inserted_slice : tensor<64x64xf32>
    }
    return %0 : tensor<64x64xf32>
  }
  func.func @_attn_fwd_scope_2_outlined_vf_0(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %1 = vector.transfer_read %arg2[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %2 = arith.subf %0, %1 : vector<64xf32>
    %3 = math.exp %2 : vector<64xf32>
    %4 = vector.transfer_write %3, %arg5[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    %5 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %6 = vector.transfer_read %arg3[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %7 = arith.mulf %5, %3 : vector<64xf32>
    %8 = arith.addf %7, %6 : vector<64xf32>
    %9 = vector.transfer_write %8, %arg4[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    return %9, %4 : tensor<64xf32>, tensor<64xf32>
  }
  func.func @_attn_fwd_scope_3_outlined_manual_vf_0(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<1x64xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: tensor<4x64x16xf16>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>) -> (tensor<4x64x16xf16>, tensor<64xf32>, tensor<64xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x64xf32>
    %1 = vector.constant_mask [1] : vector<64xi1>
    %2 = tensor.empty() : tensor<1xf32>
    %3:3 = scf.for %arg9 = %arg3 to %arg4 step %arg5 iter_args(%arg10 = %arg6, %arg11 = %arg7, %arg12 = %arg8) -> (tensor<4x64x16xf16>, tensor<64xf32>, tensor<64xf32>)  : i32 {
      %4 = arith.index_cast %arg9 : i32 to index
      %extracted_slice = tensor.extract_slice %arg0[%4, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%4] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %5 = vector.transfer_read %extracted_slice[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %6 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %7 = arith.mulf %5, %6 : vector<1x64xf32>
      %8 = vector.transfer_read %2[%c0], %cst {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
      %9 = vector.multi_reduction <maximumf>, %7, %8 [1] : vector<1x64xf32> to vector<1xf32>
      %10 = vector.transfer_read %extracted_slice_0[%c0], %cst, %1 {in_bounds = [true]} : tensor<1xf32>, vector<64xf32>
      %11 = vector.broadcast %9 : vector<1xf32> to vector<64xf32>
      %12 = arith.maximumf %10, %11 : vector<64xf32>
      %extracted_slice_1 = tensor.extract_slice %arg12[%4] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %13 = vector.transfer_write %12, %extracted_slice_1[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<1xf32>
      %14 = vector.transfer_read %13[%c0], %cst {in_bounds = [true, true], permutation_map = #map} : tensor<1xf32>, vector<1x64xf32>
      %15 = arith.subf %7, %14 : vector<1x64xf32>
      %16 = math.exp %15 : vector<1x64xf32>
      %17 = vector.transfer_write %16, %0[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
      %expanded = tensor.expand_shape %17 [[0], [1, 2, 3]] output_shape [1, 4, 1, 16] : tensor<1x64xf32> into tensor<1x4x1x16xf32>
      %collapsed = tensor.collapse_shape %expanded [[0, 1], [2], [3]] : tensor<1x4x1x16xf32> into tensor<4x1x16xf32>
      %18 = vector.transfer_read %collapsed[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : tensor<4x1x16xf32>, vector<4x1x16xf32>
      %19 = arith.truncf %18 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<4x1x16xf32> to vector<4x1x16xf16>
      %extracted_slice_2 = tensor.extract_slice %arg10[0, %4, 0] [4, 1, 16] [1, 1, 1] : tensor<4x64x16xf16> to tensor<4x1x16xf16>
      %20 = vector.transfer_write %19, %extracted_slice_2[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<4x1x16xf16>, tensor<4x1x16xf16>
      %inserted_slice = tensor.insert_slice %20 into %arg10[0, %4, 0] [4, 1, 16] [1, 1, 1] : tensor<4x1x16xf16> into tensor<4x64x16xf16>
      %21 = vector.multi_reduction <add>, %16, %8 [1] : vector<1x64xf32> to vector<1xf32>
      %extracted_slice_3 = tensor.extract_slice %arg11[%4] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %22 = vector.transfer_write %21, %extracted_slice_3[%c0] {in_bounds = [true]} : vector<1xf32>, tensor<1xf32>
      %inserted_slice_4 = tensor.insert_slice %22 into %arg11[%4] [1] [1] : tensor<1xf32> into tensor<64xf32>
      %inserted_slice_5 = tensor.insert_slice %13 into %arg12[%4] [1] [1] : tensor<1xf32> into tensor<64xf32>
      scf.yield %inserted_slice, %inserted_slice_4, %inserted_slice_5 : tensor<4x64x16xf16>, tensor<64xf32>, tensor<64xf32>
    }
    return %3#0, %3#1, %3#2 : tensor<4x64x16xf16>, tensor<64xf32>, tensor<64xf32>
  }
  func.func @_attn_fwd_outlined_vf_4(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg2 = %c0 to %c64 step %c1 iter_args(%arg3 = %arg1) -> (tensor<64x64xf16>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg3[%arg2, 0] [1, 64] [1, 1] : tensor<64x64xf16> to tensor<1x64xf16>
      %1 = vector.transfer_read %extracted_slice[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %2 = arith.truncf %1 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
      %3 = vector.transfer_write %2, %extracted_slice_0[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf16>, tensor<1x64xf16>
      %inserted_slice = tensor.insert_slice %3 into %arg3[%arg2, 0] [1, 64] [1, 1] : tensor<1x64xf16> into tensor<64x64xf16>
      scf.yield %inserted_slice : tensor<64x64xf16>
    }
    return %0 : tensor<64x64xf16>
  }
  func.func @_attn_fwd_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf16> {tt.divisibility = 16 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32}, %arg10: memref<?xi32> {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, true, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c4096_i64 = arith.constant 4096 : i64
    %c15_i32 = arith.constant 15 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c4_i64 = arith.constant 4 : i64
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c524288_i64 = arith.constant 524288 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg11, %arg12 : i32
    %1 = arith.muli %0, %arg13 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg13, %arg12 : i32
    %5 = arith.divsi %3, %4 : i32
    %6 = arith.remsi %5, %arg11 : i32
    %7 = tensor.empty() : tensor<1x64xf32>
    %8 = tensor.empty() : tensor<64xf32>
    %9 = tensor.empty() : tensor<64x64xf32>
    %10 = tensor.empty() : tensor<64xf32>
    %11:4 = call @_attn_fwd_outlined_merged_merged_vf_0(%7, %8, %9, %10) {hivm.vector_function, no_inline, ptc_simdvf} : (tensor<1x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>) -> (tensor<1x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>)
    %12 = hivm.hir.get_sub_block_idx -> i64
    %13 = arith.muli %12, %c4_i64 : i64
    %14 = arith.index_cast %13 : i64 to index
    %15 = arith.muli %12, %c4096_i64 : i64
    %16 = arith.index_cast %12 : i64 to index
    %17 = arith.cmpi eq, %16, %c0 : index
    scf.for %arg14 = %6 to %c8192_i32 step %c32_i32  : i32 {
      %18 = arith.divsi %arg14, %c8_i32 : i32
      %19 = arith.remsi %arg14, %c8_i32 : i32
      %20 = arith.divsi %18, %c8_i32 : i32
      %21 = arith.remsi %18, %c8_i32 : i32
      %22 = arith.extsi %20 : i32 to i64
      %23 = arith.muli %22, %c524288_i64 : i64
      %24 = arith.extsi %21 : i32 to i64
      %25 = arith.muli %24, %c65536_i64 : i64
      %26 = arith.addi %23, %25 : i64
      %27 = arith.index_cast %26 : i64 to index
      %28 = arith.muli %19, %c128_i32 : i32
      %29 = arith.index_cast %28 : i32 to index
      %30 = affine.apply #map1()[%27, %29]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%30], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
      %alloc = memref.alloc() : memref<128x64xf16>
      hivm.hir.load ins(%reinterpret_cast : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc : memref<128x64xf16>)
      %alloc_0 = memref.alloc() : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_0 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_1 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_1 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<64x64xf32, #hivm.address_space<ub>>
      %alloc_2 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_2 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<64x64xf32, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc_1 : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
      %subview = memref.subview %alloc_0[0, %14, 0, 0] [4, 4, 16, 16] [1, 1, 1, 1] : memref<4x8x16x16xf16, #hivm.address_space<cbuf>> to memref<4x4x16x16xf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
      %memspacecast_3 = memref.memory_space_cast %alloc_2 : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
      %31:4 = scf.for %arg15 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg16 = %11#1, %arg17 = %11#3, %arg18 = %11#2, %arg19 = %c0_i32) -> (tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>, i32)  : i32 {
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
        %37 = bufferization.to_tensor %memspacecast restrict writable : memref<64x64xf32>
        %alloc_5 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_5 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<64xf32, #hivm.address_space<ub>>
        %memspacecast_6 = memref.memory_space_cast %alloc_5 : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32>
        %38 = bufferization.to_tensor %memspacecast_6 restrict writable : memref<64xf32>
        %alloc_7 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_7 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<64xf32, #hivm.address_space<ub>>
        %memspacecast_8 = memref.memory_space_cast %alloc_7 : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32>
        %39 = bufferization.to_tensor %memspacecast_8 restrict writable : memref<64xf32>
        %alloc_9 = memref.alloc() : memref<4x64x16xf16, #hivm.address_space<ub>>
        annotation.mark %alloc_9 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>} : memref<4x64x16xf16, #hivm.address_space<ub>>
        %memspacecast_10 = memref.memory_space_cast %alloc_9 : memref<4x64x16xf16, #hivm.address_space<ub>> to memref<4x64x16xf16>
        %40 = bufferization.to_tensor %memspacecast_10 restrict writable : memref<4x64x16xf16>
        %41:3 = func.call @_attn_fwd_scope_3_outlined_manual_vf_0(%37, %arg17, %11#0, %c0_i32, %c64_i32, %c1_i32, %40, %38, %39) {hivm.vector_function} : (tensor<64x64xf32>, tensor<64xf32>, tensor<1x64xf32>, i32, i32, i32, tensor<4x64x16xf16>, tensor<64xf32>, tensor<64xf32>) -> (tensor<4x64x16xf16>, tensor<64xf32>, tensor<64xf32>)
        %expanded = tensor.expand_shape %41#0 [[0], [1, 2], [3]] output_shape [4, 4, 16, 16] : tensor<4x64x16xf16> into tensor<4x4x16x16xf16>
        %42 = bufferization.to_memref %expanded : memref<4x4x16x16xf16, #hivm.address_space<ub>>
        hivm.hir.copy ins(%42 : memref<4x4x16x16xf16, #hivm.address_space<ub>>) outs(%subview : memref<4x4x16x16xf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>)
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        %43 = tensor.empty() : tensor<64xf32>
        %44 = tensor.empty() : tensor<64xf32>
        %45:2 = func.call @_attn_fwd_scope_2_outlined_vf_0(%arg16, %arg17, %41#2, %41#1, %43, %44) {hivm.vector_function, no_inline} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>)
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
        %46 = bufferization.to_tensor %memspacecast_3 restrict writable : memref<64x64xf32>
        %47 = arith.cmpi slt, %arg15, %c15_i32 : i32
        %48 = scf.if %47 -> (tensor<64x64xf32>) {
          %50 = func.call @_attn_fwd_scope_1_outlined_manual_vf_0(%46, %45#1, %c0_i32, %c64_i32, %c1_i32, %arg18) {hivm.vector_function} : (tensor<64x64xf32>, tensor<64xf32>, i32, i32, i32, tensor<64x64xf32>) -> tensor<64x64xf32>
          scf.yield %50 : tensor<64x64xf32>
        } else {
          %50 = func.call @_attn_fwd_scope_0_outlined_manual_vf_0(%46, %45#1, %45#0, %c0_i32, %c64_i32, %c1_i32, %arg18) {hivm.vector_function} : (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, i32, i32, i32, tensor<64x64xf32>) -> tensor<64x64xf32>
          scf.yield %50 : tensor<64x64xf32>
        }
        %49 = arith.addi %arg19, %c1_i32 : i32
        scf.yield %45#0, %41#2, %48, %49 : tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>, i32
      }
      %32 = arith.addi %26, %15 : i64
      %33 = arith.index_cast %32 : i64 to index
      %34 = affine.apply #map1()[%33, %29]
      %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%34], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
      %35 = tensor.empty() : tensor<64x64xf16>
      %36 = func.call @_attn_fwd_outlined_vf_4(%31#2, %35) {hivm.vector_function, no_inline} : (tensor<64x64xf32>, tensor<64x64xf16>) -> tensor<64x64xf16>
      scf.if %17 {
        hivm.hir.store ins(%36 : tensor<64x64xf16>) outs(%reinterpret_cast_4 : memref<64x64xf16, strided<[64, 1], offset: ?>>)
      } {limit_sub_block_id0}
    }
    return
  }
}

// -----
module {
  func.func @_hstu_attn_bwd_mix_aiv_outlined_merged_vf_0(%arg0: tensor<1x16x1x16xf16>, %arg1: tensor<1x16x1x16xf16>, %arg2: tensor<1x1x16x16xf16>, %arg3: tensor<1x1x16x16xf16>, %arg4: tensor<16x128xf32>, %arg5: tensor<16x128xf16>, %arg6: tensor<16x128xf16>, %arg7: tensor<16x128xf16>) -> (tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, tensor<16x128xf16>, tensor<16x128xf16>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    return %arg2, %arg3, %arg6, %arg7 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, tensor<16x128xf16>, tensor<16x128xf16>
  }
  func.func @move_before_maintain_order(%arg0: memref<?xf16>, %arg1: memref<?xf16>, %arg2: i32, %arg3: i32, %arg4: tensor<1x16x1x16xf16>, %arg5: tensor<1x16x1x16xf16>, %arg6: tensor<1x1x16x16xf16>, %arg7: tensor<1x1x16x16xf16>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %0 = arith.muli %arg3, %arg2 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [16, 128], strides: [%c128, 1] : memref<?xf16> to memref<16x128xf16, strided<[?, 1], offset: ?>>
    %subview = memref.subview %reinterpret_cast[0, 0] [%c16, 128] [1, 1] : memref<16x128xf16, strided<[?, 1], offset: ?>> to memref<?x128xf16, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc() : memref<16x128xf32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<16x128xf32>
    %alloc_0 = memref.alloc() : memref<16x128xf16>
    %3 = bufferization.to_tensor %alloc_0 restrict writable : memref<16x128xf16>
    %4 = tensor.empty() : tensor<16x128xf16>
    %5 = tensor.empty() : tensor<16x128xf16>
    %6:4 = call @_hstu_attn_bwd_mix_aiv_outlined_merged_vf_0(%arg4, %arg5, %arg6, %arg7, %2, %3, %4, %5) {hivm.vector_function, no_inline, ptc_simdvf} : (tensor<1x16x1x16xf16>, tensor<1x16x1x16xf16>, tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, tensor<16x128xf32>, tensor<16x128xf16>, tensor<16x128xf16>, tensor<16x128xf16>) -> (tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, tensor<16x128xf16>, tensor<16x128xf16>)
    return
  }
}

// -----
module {
  func.func @_hstu_attn_bwd_mix_aiv_outlined_vf_0(%arg0: tensor<16x128xf32>, %arg1: tensor<16x128xf16>, %arg2: tensor<16x128xf16>, %arg3: tensor<16x128xf16>) -> (tensor<16x128xf16>, tensor<16x128xf16>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    return %arg2, %arg3 : tensor<16x128xf16>, tensor<16x128xf16>
  }
  func.func @_hstu_attn_bwd_mix_aiv_outlined_vf_4(%arg0: tensor<1x16x1x16xf16>, %arg1: tensor<1x16x1x16xf16>, %arg2: tensor<1x1x16x16xf16>, %arg3: tensor<1x1x16x16xf16>) -> (tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    return %arg2, %arg3 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>
  }
  func.func @not_merge_when_has_sync_in_between(%arg0: memref<?xf16>, %arg1: memref<?xf16>, %arg2: i32, %arg3: i32, %arg4: tensor<1x16x1x16xf16>, %arg5: tensor<1x16x1x16xf16>, %arg6: tensor<1x1x16x16xf16>, %arg7: tensor<1x1x16x16xf16>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %0:2 = call @_hstu_attn_bwd_mix_aiv_outlined_vf_4(%arg4, %arg5, %arg6, %arg7) {hivm.vector_function, no_inline} : (tensor<1x16x1x16xf16>, tensor<1x16x1x16xf16>, tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>) -> (tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>)
    %1 = arith.muli %arg3, %arg2 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%2], sizes: [16, 128], strides: [%c128, 1] : memref<?xf16> to memref<16x128xf16, strided<[?, 1], offset: ?>>
    %subview = memref.subview %reinterpret_cast[0, 0] [%c16, 128] [1, 1] : memref<16x128xf16, strided<[?, 1], offset: ?>> to memref<?x128xf16, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc() : memref<16x128xf32>
    %3 = bufferization.to_tensor %alloc restrict writable : memref<16x128xf32>
    %alloc_0 = memref.alloc() : memref<16x128xf16>
    %4 = bufferization.to_tensor %alloc_0 restrict writable : memref<16x128xf16>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
    %5 = tensor.empty() : tensor<16x128xf16>
    %6 = tensor.empty() : tensor<16x128xf16>
    %7:2 = call @_hstu_attn_bwd_mix_aiv_outlined_vf_0(%3, %4, %5, %6) {hivm.vector_function, no_inline} : (tensor<16x128xf32>, tensor<16x128xf16>, tensor<16x128xf16>, tensor<16x128xf16>) -> (tensor<16x128xf16>, tensor<16x128xf16>)
    return
  }
}

// -----
module {
  func.func @_hstu_attn_bwd_mix_aiv_outlined_merged_vf_0(%arg0: tensor<1x16x1x16xf16>, %arg1: tensor<1x16x1x16xf16>, %arg2: tensor<1x1x16x16xf16>, %arg3: tensor<1x1x16x16xf16>, %arg4: tensor<16x128xf32>, %arg5: tensor<16x128xf16>, %arg6: tensor<16x128xf16>, %arg7: tensor<16x128xf16>) -> (tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, tensor<16x128xf16>, tensor<16x128xf16>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    return %arg2, %arg3, %arg6, %arg7 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, tensor<16x128xf16>, tensor<16x128xf16>
  }
  func.func @merge_when_node_with_block_in_between(%arg0: memref<?xf16>, %arg1: memref<?xf16>, %arg2: i32, %arg3: i32, %arg4: tensor<1x16x1x16xf16>, %arg5: tensor<1x16x1x16xf16>, %arg6: tensor<1x1x16x16xf16>, %arg7: tensor<1x1x16x16xf16>, %arg8: i1, %arg9: memref<1x1x16x16xf16>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %0 = arith.muli %arg3, %arg2 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [16, 128], strides: [%c128, 1] : memref<?xf16> to memref<16x128xf16, strided<[?, 1], offset: ?>>
    %subview = memref.subview %reinterpret_cast[0, 0] [%c16, 128] [1, 1] : memref<16x128xf16, strided<[?, 1], offset: ?>> to memref<?x128xf16, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc() : memref<16x128xf32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<16x128xf32>
    %alloc_0 = memref.alloc() : memref<16x128xf16>
    %3 = bufferization.to_tensor %alloc_0 restrict writable : memref<16x128xf16>
    %4 = tensor.empty() : tensor<16x128xf16>
    %5 = tensor.empty() : tensor<16x128xf16>
    %6:4 = call @_hstu_attn_bwd_mix_aiv_outlined_merged_vf_0(%arg4, %arg5, %arg6, %arg7, %2, %3, %4, %5) {hivm.vector_function, no_inline, ptc_simdvf} : (tensor<1x16x1x16xf16>, tensor<1x16x1x16xf16>, tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, tensor<16x128xf32>, tensor<16x128xf16>, tensor<16x128xf16>, tensor<16x128xf16>) -> (tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, tensor<16x128xf16>, tensor<16x128xf16>)
    scf.if %arg8 {
      hivm.hir.store ins(%6#0 : tensor<1x1x16x16xf16>) outs(%arg9 : memref<1x1x16x16xf16>)
    } {limit_sub_block_id0}
    return
  }
}

// -----
#map = affine_map<(d0) -> (-d0 + 248, 64)>
module {
  func.func @kernel_outlined_merged_vf_0(%arg0: memref<8x248xf32>, %arg1: memref<8x248xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c248 = arith.constant 248 : index
    %c0 = arith.constant 0 : index
    scf.for %arg2 = %c0 to %c8 step %c1 {
      scf.for %arg3 = %c0 to %c248 step %c64 {
        %0 = affine.min #map(%arg3)
        %subview = memref.subview %arg0[%arg2, %arg3] [1, %0] [1, 1] : memref<8x248xf32> to memref<1x?xf32, strided<[248, 1], offset: ?>>
        %1 = vector.create_mask %c1, %0 : vector<1x64xi1>
        vector.transfer_write %cst, %subview[%c0, %c0], %1 {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x?xf32, strided<[248, 1], offset: ?>>
      }
    }
    %cst_0 = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %c1_1 = arith.constant 1 : index
    %c8_2 = arith.constant 8 : index
    %c64_3 = arith.constant 64 : index
    %c248_4 = arith.constant 248 : index
    %c0_5 = arith.constant 0 : index
    scf.for %arg2 = %c0_5 to %c8_2 step %c1_1 {
      scf.for %arg3 = %c0_5 to %c248_4 step %c64_3 {
        %0 = affine.min #map(%arg3)
        %subview = memref.subview %arg1[%arg2, %arg3] [1, %0] [1, 1] : memref<8x248xf32> to memref<1x?xf32, strided<[248, 1], offset: ?>>
        %1 = vector.create_mask %c1_1, %0 : vector<1x64xi1>
        vector.transfer_write %cst_0, %subview[%c0_5, %c0_5], %1 {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x?xf32, strided<[248, 1], offset: ?>>
      }
    }
    return
  }
  func.func @not_merge_when_load_write_to_same_buffer(%arg0: memref<?xf32>, %arg1: index, %arg2: memref<?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<8x248xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%arg1], sizes: [8, 248], strides: [128, 1] : memref<?xf32> to memref<8x248xf32, strided<[128, 1], offset: ?>>
    %subview = memref.subview %reinterpret_cast[0, 0] [%arg1, 128] [1, 1] : memref<8x248xf32, strided<[128, 1], offset: ?>> to memref<?x128xf32, strided<[128, 1], offset: ?>>
    %subview_0 = memref.subview %alloc[0, 0] [%arg1, 128] [1, 1] : memref<8x248xf32> to memref<?x128xf32, strided<[248, 1]>>
    %alloc_1 = memref.alloc() : memref<8x248xf32>
    call @kernel_outlined_merged_vf_0(%alloc, %alloc_1) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<8x248xf32>, memref<8x248xf32>) -> ()
    hivm.hir.load ins(%subview : memref<?x128xf32, strided<[128, 1], offset: ?>>) outs(%subview_0 : memref<?x128xf32, strided<[248, 1]>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index
    return
  }
}

// -----
module {
  func.func @vf_1(%arg0: memref<16x128xf16>) attributes {hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f16
    affine.store %cst, %arg0[0, 0] : memref<16x128xf16>
    return
  }
  func.func @vf_2(%arg0: tensor<16x128xf16>) -> tensor<16x128xf16> attributes {hivm.vector_function, no_inline} {
    return %arg0 : tensor<16x128xf16>
  }
  func.func @test_bridge_dependency(%arg0: memref<16x128xf16>) {
    call @vf_1(%arg0) {hivm.vector_function, no_inline} : (memref<16x128xf16>) -> ()
    %0 = bufferization.to_tensor %arg0 : memref<16x128xf16>
    %c1 = arith.constant 1 : index
    %1 = call @vf_2(%0) {hivm.vector_function, no_inline} : (tensor<16x128xf16>) -> tensor<16x128xf16>
    return
  }
}

// -----
module {
  func.func @vf_1(%arg0: memref<16x128xf16>) attributes {hivm.vector_function, no_inline} {
    %cst = arith.constant 1.000000e+00 : f16
    affine.store %cst, %arg0[0, 0] : memref<16x128xf16>
    return
  }
  func.func @vf_2(%arg0: tensor<16x128xf16>) -> tensor<16x128xf16> attributes {hivm.vector_function, no_inline} {
    return %arg0 : tensor<16x128xf16>
  }
  func.func @test_if_capture_bridge(%arg0: memref<16x128xf16>, %arg1: i1) {
    call @vf_1(%arg0) {hivm.vector_function, no_inline} : (memref<16x128xf16>) -> ()
    %0 = scf.if %arg1 -> (tensor<16x128xf16>) {
      %2 = bufferization.to_tensor %arg0 : memref<16x128xf16>
      scf.yield %2 : tensor<16x128xf16>
    } else {
      %2 = tensor.empty() : tensor<16x128xf16>
      scf.yield %2 : tensor<16x128xf16>
    }
    %1 = call @vf_2(%0) {hivm.vector_function, no_inline} : (tensor<16x128xf16>) -> tensor<16x128xf16>
    return
  }
}
