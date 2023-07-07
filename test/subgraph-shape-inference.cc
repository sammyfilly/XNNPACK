// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>
#include <xnnpack/subgraph.h>

#include "subgraph-tester.h"
#include <gtest/gtest.h>


namespace xnnpack {

TEST(SUBGRAPH, tensor_with_dynamic_dims_set_correctly) {
  auto tester = SubgraphTester(1);
  tester
    .AddInputTensorF32({XNN_UNKNOWN_DIM, 1}, 0);
  const xnn_value* input = tester.Value(0);
  EXPECT_FALSE(xnn_tensor_shape_is_static(input));
}

TEST(SUBGRAPH, quantized_tensor_with_dynamic_dims_set_correctly) {
  auto tester = SubgraphTester(1);
  tester
    .AddInputTensorQS8(/*zero_point=*/3, /*scale=*/1.1f, {XNN_UNKNOWN_DIM, 1}, 0);
  const xnn_value* input = tester.Value(0);
  EXPECT_FALSE(xnn_tensor_shape_is_static(input));
}

TEST(SUBGRAPH, infer_shape_for_fully_connected_node) {
  auto tester = SubgraphTester(6);
  const size_t input_channels = 3;
  const size_t output_channels = 5;
  const uint32_t input_id = 0;
  const uint32_t filter_id = 1;
  const uint32_t output_id = 2;
  tester
    .AddInputTensorF32({XNN_UNKNOWN_DIM, XNN_UNKNOWN_DIM}, input_id)
    .AddStaticTensorF32({output_channels, input_channels}, TensorType::kDense, filter_id)
    .AddOutputTensorF32({XNN_UNKNOWN_DIM, XNN_UNKNOWN_DIM}, output_id)
    .AddFullyConnected(input_id, filter_id, XNN_INVALID_VALUE_ID, output_id)
    .InferShape()
  ;

  const xnn_value* input = tester.Value(input_id);
  // Batch size cannot be inferred.
  ASSERT_EQ(input->shape.dim[0], XNN_UNKNOWN_DIM);
  EXPECT_EQ(input->shape.dim[1], input_channels);

  const xnn_value* output = tester.Value(output_id);
  // Batch size cannot be inferred.
  ASSERT_EQ(output->shape.dim[0], XNN_UNKNOWN_DIM);
  EXPECT_EQ(output->shape.dim[1], output_channels);

  // Values are still dynamic after inference.
  EXPECT_FALSE(xnn_tensor_shape_is_static(input));
  EXPECT_FALSE(xnn_tensor_shape_is_static(output));
}

TEST(SUBGRAPH, infer_shape_for_fully_connected_node_more_dims) {
  auto tester = SubgraphTester(6);
  const size_t batch1_dim = 3;
  const size_t batch2_dim = 5;
  const size_t input_channels = 7;
  const size_t output_channels = 9;
  const uint32_t input_id = 0;
  const uint32_t filter_id = 1;
  const uint32_t output_id = 2;
  tester
    .AddInputTensorF32({batch1_dim, batch2_dim, XNN_UNKNOWN_DIM}, input_id)
    .AddStaticTensorF32({output_channels, input_channels}, TensorType::kDense, filter_id)
    .AddOutputTensorF32({batch1_dim, batch2_dim, XNN_UNKNOWN_DIM}, output_id)
    .AddFullyConnected(input_id, filter_id, XNN_INVALID_VALUE_ID, output_id)
    .InferShape()
  ;

  const xnn_value* input = tester.Value(input_id);
  ASSERT_EQ(input->shape.dim[2], input_channels);

  const xnn_value* output = tester.Value(output_id);
  ASSERT_EQ(output->shape.dim[2], output_channels);

  // Values are still dynamic after inference.
  EXPECT_FALSE(xnn_tensor_shape_is_static(input));
  EXPECT_FALSE(xnn_tensor_shape_is_static(output));
}

}  // namespace xnnpack
