// SPDX-License-Identifier: LGPL-3.0-or-later
// Regression test for the full-node packed batch API, including per-frame
// local/ghost boundaries. Generate the model with
// source/tests/infer/gen_dpa4c_canonical_batch.py.
#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <vector>

#include "DeepPot.h"
#include "DeepPotPTExpt.h"

#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT
#include <torch/torch.h>

#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 3)
namespace {
constexpr const char* kModelPath =
    "../../tests/infer/deeppot_dpa4c_canonical_batch.pt2";

struct Frame {
  std::vector<std::int64_t> atype;
  std::vector<float> coord;
  int n_local;
};

struct CanonicalGraph {
  std::vector<std::int64_t> atype;
  std::vector<std::uint32_t> source;
  std::vector<float> edge_vec;
  std::vector<std::int64_t> destination_row_ptr{0};
  std::vector<std::int64_t> source_row_ptr{0};
  std::vector<std::uint32_t> source_order;
  std::vector<std::int64_t> n_node;
  std::vector<std::int64_t> n_local;
};

CanonicalGraph make_graph(const std::vector<Frame>& frames) {
  CanonicalGraph graph;
  std::int64_t node_offset = 0;
  for (const Frame& frame : frames) {
    const std::int64_t n = frame.atype.size();
    graph.n_node.push_back(n);
    graph.n_local.push_back(frame.n_local);
    graph.atype.insert(graph.atype.end(), frame.atype.begin(), frame.atype.end());
    for (std::int64_t dst = 0; dst < n; ++dst) {
      for (std::int64_t src = 0; src < n; ++src) {
        if (src == dst) {
          continue;
        }
        graph.source.push_back(static_cast<std::uint32_t>(node_offset + src));
        for (int d = 0; d < 3; ++d) {
          graph.edge_vec.push_back(frame.coord[3 * src + d] -
                                   frame.coord[3 * dst + d]);
        }
      }
      graph.destination_row_ptr.push_back(graph.source.size());
    }
    node_offset += n;
  }

  for (std::uint32_t src = 0; src < graph.atype.size(); ++src) {
    for (std::uint32_t edge = 0; edge < graph.source.size(); ++edge) {
      if (graph.source[edge] == src) {
        graph.source_order.push_back(edge);
      }
    }
    graph.source_row_ptr.push_back(graph.source_order.size());
  }
  return graph;
}

template <typename T>
torch::Tensor to_cuda(const std::vector<T>& values,
                      const at::ScalarType dtype) {
  auto host = torch::from_blob(
                  const_cast<T*>(values.data()),
                  {static_cast<std::int64_t>(values.size())},
                  torch::TensorOptions().dtype(dtype))
                  .clone();
  return host.to(torch::Device(torch::kCUDA, 0));
}

struct DeviceGraph {
  torch::Tensor atype;
  torch::Tensor source;
  torch::Tensor edge_vec;
  torch::Tensor destination_row_ptr;
  torch::Tensor source_row_ptr;
  torch::Tensor source_order;
  torch::Tensor n_node;
  torch::Tensor n_local;
};

DeviceGraph upload(const CanonicalGraph& graph) {
  return {to_cuda(graph.atype, torch::kInt64),
          to_cuda(graph.source, torch::kUInt32),
          to_cuda(graph.edge_vec, torch::kFloat32),
          to_cuda(graph.destination_row_ptr, torch::kInt64),
          to_cuda(graph.source_row_ptr, torch::kInt64),
          to_cuda(graph.source_order, torch::kUInt32),
          to_cuda(graph.n_node, torch::kInt64),
          to_cuda(graph.n_local, torch::kInt64)};
}

const auto kOutputOptions = torch::TensorOptions()
                                .dtype(torch::kFloat64)
                                .device(torch::Device(torch::kCUDA, 0));

}  // namespace
#endif
#endif

TEST(TestDeepPotDpa4cCanonicalBatchPtExpt, MatchesFramesAndGhostContract) {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT
  GTEST_SKIP() << "PyTorch Exportable support is not enabled";
#elif TORCH_VERSION_MAJOR < 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 3)
  GTEST_SKIP() << "canonical uint32 indices require PyTorch 2.3 or later";
#else
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is unavailable";
  }
  std::ifstream model_file(kModelPath);
  if (!model_file.good()) {
    GTEST_SKIP() << "Generate the fixture with "
                    "source/tests/infer/gen_dpa4c_canonical_batch.py";
  }

  deepmd::DeepPot dp;
  dp.init(kModelPath);
  ASSERT_TRUE(dp.uses_canonical_graph_inference());

  const std::vector<std::vector<Frame>> scenarios = {
      {{{0, 1, 0}, {0.f, 0.f, 0.f, 1.1f, 0.2f, 0.1f, 0.2f, 1.2f, 0.3f}, 3},
       {{1, 0, 1, 0},
        {0.1f, 0.1f, 0.f, 1.2f, 0.2f, 0.1f, 0.2f, 1.3f, 0.2f,
         1.1f, 1.2f, 0.4f},
        4}},
      {{{0, 1, 0}, {0.f, 0.f, 0.f, 1.1f, 0.2f, 0.1f, 0.2f, 1.2f, 0.3f}, 2},
       {{1, 0, 1, 0},
        {0.1f, 0.1f, 0.f, 1.2f, 0.2f, 0.1f, 0.2f, 1.3f, 0.2f,
         1.1f, 1.2f, 0.4f},
        3}}};
  constexpr double sentinel = 123456.75;
  for (std::size_t scenario_id = 0; scenario_id < scenarios.size();
       ++scenario_id) {
    SCOPED_TRACE(scenario_id == 0 ? "all-local frames" : "frames with ghosts");
    const auto& frames = scenarios[scenario_id];
    const CanonicalGraph batch_graph = make_graph(frames);
    const DeviceGraph batch = upload(batch_graph);
    const int nall_nodes = static_cast<int>(batch_graph.atype.size());
    const std::int64_t edge_storage = batch_graph.source.size();
    auto batch_energy = torch::full({nall_nodes + 1}, sentinel, kOutputOptions);
    auto batch_force = torch::empty({nall_nodes * 3}, kOutputOptions);
    auto batch_virial = torch::empty({nall_nodes * 9}, kOutputOptions);

    dp.compute_canonical_graph_gpu_batch(
        batch_energy.data_ptr<double>(), batch_force.data_ptr<double>(),
        batch_virial.data_ptr<double>(), batch.atype.data_ptr<std::int64_t>(),
        batch.source.data_ptr<std::uint32_t>(), batch.edge_vec.data_ptr<float>(),
        batch.destination_row_ptr.data_ptr<std::int64_t>(),
        batch.source_row_ptr.data_ptr<std::int64_t>(),
        batch.source_order.data_ptr<std::uint32_t>(),
        batch.n_node.data_ptr<std::int64_t>(),
        batch.n_local.data_ptr<std::int64_t>(), static_cast<int>(frames.size()),
        nall_nodes, edge_storage);

    auto batch_energy_cpu = batch_energy.cpu();
    auto batch_force_cpu = batch_force.cpu();
    auto batch_virial_cpu = batch_virial.cpu();
    std::int64_t node_offset = 0;
    for (const Frame& frame : frames) {
      const CanonicalGraph one_graph = make_graph({frame});
      const DeviceGraph one = upload(one_graph);
      const int n = static_cast<int>(frame.atype.size());
      const auto one_energy = torch::empty({frame.n_local}, kOutputOptions);
      const auto one_force = torch::empty({n * 3}, kOutputOptions);
      const auto one_virial = torch::empty({n * 9}, kOutputOptions);
      dp.compute_canonical_graph_gpu(
          one_energy.data_ptr<double>(), one_force.data_ptr<double>(),
          one_virial.data_ptr<double>(), one.atype.data_ptr<std::int64_t>(),
          one.source.data_ptr<std::uint32_t>(), one.edge_vec.data_ptr<float>(),
          one.destination_row_ptr.data_ptr<std::int64_t>(),
          one.source_row_ptr.data_ptr<std::int64_t>(),
          one.source_order.data_ptr<std::uint32_t>(), frame.n_local, n,
          static_cast<std::int64_t>(one_graph.source.size()));

      EXPECT_TRUE(torch::allclose(
          batch_energy_cpu.slice(0, node_offset, node_offset + frame.n_local),
          one_energy.cpu(), 1e-5, 1e-5));
      EXPECT_TRUE(torch::equal(
          batch_energy_cpu.slice(0, node_offset + frame.n_local, node_offset + n),
          torch::zeros({n - frame.n_local}, torch::kFloat64)));
      EXPECT_TRUE(torch::allclose(
          batch_force_cpu.slice(0, node_offset * 3, (node_offset + n) * 3),
          one_force.cpu(), 1e-5, 1e-5));
      EXPECT_TRUE(torch::allclose(
          batch_virial_cpu.slice(0, node_offset * 9, (node_offset + n) * 9),
          one_virial.cpu(), 1e-5, 1e-5));
      node_offset += n;
    }
    EXPECT_DOUBLE_EQ(batch_energy_cpu[nall_nodes].item<double>(), sentinel);
  }
#endif
}
