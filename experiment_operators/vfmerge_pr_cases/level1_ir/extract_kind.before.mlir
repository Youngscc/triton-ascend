module {
  func.func @extract_mismatch_vf_0(%arg0: tensor<1xf32>) -> tensor<1xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    return %arg0 : tensor<1xf32>
  }
  func.func @extract_mismatch_vf_1(%arg0: tensor<32x128xf32>) -> tensor<32x128xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    return %arg0 : tensor<32x128xf32>
  }
  func.func @extract_mismatch_main(%arg0: tensor<1xf32>, %arg1: tensor<32x128xf32>) -> (f32, tensor<32x128xf32>) {
    %c0 = arith.constant 0 : index
    %0 = call @extract_mismatch_vf_0(%arg0) {hivm.vector_function, no_inline} : (tensor<1xf32>) -> tensor<1xf32>
    %extracted = tensor.extract %0[%c0] : tensor<1xf32>
    %1 = call @extract_mismatch_vf_1(%arg1) {hivm.vector_function, no_inline} : (tensor<32x128xf32>) -> tensor<32x128xf32>
    return %extracted, %1 : f32, tensor<32x128xf32>
  }
}

// -----
module {
  func.func @same_extract_vf_0(%arg0: tensor<1xf32>) -> tensor<1xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    return %arg0 : tensor<1xf32>
  }
  func.func @same_extract_vf_1(%arg0: tensor<1xf32>) -> tensor<1xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    return %arg0 : tensor<1xf32>
  }
  func.func @same_extract_main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> (f32, f32) {
    %c0 = arith.constant 0 : index
    %0 = call @same_extract_vf_0(%arg0) {hivm.vector_function, no_inline} : (tensor<1xf32>) -> tensor<1xf32>
    %extracted = tensor.extract %0[%c0] : tensor<1xf32>
    %1 = call @same_extract_vf_1(%arg1) {hivm.vector_function, no_inline} : (tensor<1xf32>) -> tensor<1xf32>
    %extracted_0 = tensor.extract %1[%c0] : tensor<1xf32>
    return %extracted, %extracted_0 : f32, f32
  }
}

// -----
module {
  func.func @same_no_extract_vf_0(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    return %arg0 : tensor<4xf32>
  }
  func.func @same_no_extract_vf_1(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    return %arg0 : tensor<4xf32>
  }
  func.func @same_no_extract_main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %0 = call @same_no_extract_vf_0(%arg0) {hivm.vector_function, no_inline} : (tensor<4xf32>) -> tensor<4xf32>
    %1 = call @same_no_extract_vf_1(%arg1) {hivm.vector_function, no_inline} : (tensor<4xf32>) -> tensor<4xf32>
    return %0, %1 : tensor<4xf32>, tensor<4xf32>
  }
}
