module {
  func.func @dependency_update_producer_vf_0(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    return %arg0 : tensor<4xf32>
  }
  func.func @dependency_update_zero_vf_0(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : tensor<4xf32>
    return %cst : tensor<4xf32>
  }
  func.func @dependency_update_reduce_a_vf_0(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    return %arg0 : tensor<4xf32>
  }
  func.func @dependency_update_reduce_b_vf_0(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    return %arg0 : tensor<4xf32>
  }
  func.func @dependency_update_main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %0 = call @dependency_update_producer_vf_0(%arg0) {hivm.vector_function, no_inline} : (tensor<4xf32>) -> tensor<4xf32>
    %1 = call @dependency_update_zero_vf_0(%arg1) {hivm.vector_function, no_inline} : (tensor<4xf32>) -> tensor<4xf32>
    %2 = call @dependency_update_reduce_a_vf_0(%1) {hivm.vector_function, no_inline} : (tensor<4xf32>) -> tensor<4xf32>
    %3 = call @dependency_update_reduce_b_vf_0(%1) {hivm.vector_function, no_inline} : (tensor<4xf32>) -> tensor<4xf32>
    return %0, %2, %3 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }
}
