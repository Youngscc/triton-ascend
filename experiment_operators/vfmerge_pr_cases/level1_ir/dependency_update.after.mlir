module {
  func.func @dependency_update_reduce_a_merged_vf_0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    return %arg0, %arg1 : tensor<4xf32>, tensor<4xf32>
  }
  func.func @dependency_update_producer_merged_vf_0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant dense<0.000000e+00> : tensor<4xf32>
    return %arg0, %cst : tensor<4xf32>, tensor<4xf32>
  }
  func.func @dependency_update_main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %0:2 = call @dependency_update_producer_merged_vf_0(%arg0, %arg1) {hivm.vector_function, no_inline, ptc_simdvf} : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
    %1:2 = call @dependency_update_reduce_a_merged_vf_0(%0#1, %0#1) {hivm.vector_function, no_inline, ptc_simdvf} : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
    return %0#0, %1#0, %1#1 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }
}
