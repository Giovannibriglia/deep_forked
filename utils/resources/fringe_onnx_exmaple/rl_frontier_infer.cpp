#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

enum class DatasetType { HASHED, MAPPED, BITMASK };
enum class CombineMode { Merged, Separated };

struct CandidateGraphTensor {
  int64_t feat_dim = 1;
  std::vector<float> node_features;      // [N, F] flattened
  std::vector<int64_t> edge_src_local;   // [E]
  std::vector<int64_t> edge_dst_local;   // [E]
  std::vector<int64_t> edge_attr;        // [E]
  std::vector<std::string> node_uid;     // [N], used in merged mode
};

struct CombinedFrontierGraphTensor {
  int64_t feat_dim = 1;
  std::vector<int64_t> node_features;      // [N, F]
  std::vector<int64_t> edge_src;         // [E]
  std::vector<int64_t> edge_dst;         // [E]
  std::vector<int64_t> edge_attr;        // [E]
  std::vector<int64_t> membership;       // [N]
  std::vector<int64_t> pool_node_index;  // optional merged helper
  std::vector<int64_t> pool_membership;  // optional merged helper
};

struct GoalGraphTensor {
  int64_t feat_dim = 1;
  std::vector<float> node_features;      // [GN, F]
  std::vector<int64_t> edge_src;         // [GE]
  std::vector<int64_t> edge_dst;         // [GE]
  std::vector<int64_t> edge_attr;        // [GE]
};

struct DotEdge {
  std::string src;
  std::string dst;
  int64_t label = 0;
};

struct DotGraph {
  std::vector<std::string> nodes;
  std::vector<DotEdge> edges;
};

static std::string trim(const std::string &s) {
  const size_t start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return "";
  }
  const size_t end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

static std::string to_upper(std::string s) {
  for (char &ch : s) {
    ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  }
  return s;
}

static std::string strip_quotes(std::string s) {
  s = trim(s);
  if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
    s = s.substr(1, s.size() - 2);
  }
  return s;
}

static DatasetType parse_dataset_type(const std::string &s) {
  const std::string u = to_upper(s);
  if (u == "HASHED") return DatasetType::HASHED;
  if (u == "MAPPED") return DatasetType::MAPPED;
  if (u == "BITMASK") return DatasetType::BITMASK;
  throw std::runtime_error("Unsupported --dataset-type: " + s);
}

static CombineMode parse_mode(const std::string &s) {
  const std::string u = to_upper(s);
  if (u == "MERGED") return CombineMode::Merged;
  if (u == "SEPARATED") return CombineMode::Separated;
  throw std::runtime_error("Unsupported --mode: " + s);
}

static std::vector<std::string> read_nonempty_lines(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open frontier list file: " + path);
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    const std::string t = trim(line);
    if (t.empty() || t.rfind("#", 0) == 0) {
      continue;
    }
    lines.push_back(t);
  }
  return lines;
}

static DotGraph parse_dot_file(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open DOT file: " + path);
  }

  DotGraph g;
  std::unordered_map<std::string, int64_t> seen_nodes;
  auto ensure_node = [&](const std::string &raw) {
    const std::string node = strip_quotes(raw);
    if (node.empty()) return;
    if (seen_nodes.find(node) == seen_nodes.end()) {
      seen_nodes.emplace(node, static_cast<int64_t>(g.nodes.size()));
      g.nodes.push_back(node);
    }
  };

  const std::regex edge_re(
      R"DOT(^\s*"?([^"\s]+)"?\s*->\s*"?([^"\s]+)"?\s*(?:\[(.*)\])?\s*;?\s*$)DOT");
  const std::regex node_re(
      R"DOT(^\s*"?([^"\s]+)"?\s*(?:\[[^\]]*\])?\s*;?\s*$)DOT");
  const std::regex label_re(R"DOT(label\s*=\s*"?(-?\d+)"?)DOT");

  std::string line;
  while (std::getline(in, line)) {
    const size_t cpos = line.find("//");
    if (cpos != std::string::npos) {
      line = line.substr(0, cpos);
    }

    const std::string t = trim(line);
    if (t.empty()) continue;
    if (t == "{" || t == "}") continue;

    const std::string tu = to_upper(t);
    if (tu.rfind("DIGRAPH", 0) == 0 || tu.rfind("GRAPH", 0) == 0 ||
        tu.rfind("SUBGRAPH", 0) == 0) {
      continue;
    }
    if (t.rfind("#", 0) == 0) continue;

    std::smatch m;
    if (std::regex_match(t, m, edge_re)) {
      const std::string src = strip_quotes(m[1].str());
      const std::string dst = strip_quotes(m[2].str());
      const std::string attrs = (m.size() >= 4) ? m[3].str() : "";
      if (src.empty() || dst.empty()) {
        throw std::runtime_error("Malformed edge line in " + path + ": " + t);
      }

      int64_t label = 0;
      std::smatch lm;
      if (std::regex_search(attrs, lm, label_re)) {
        label = std::stoll(lm[1].str());
      } else {
        throw std::runtime_error(
            "Edge without integer label in " + path + ": " + t);
      }

      ensure_node(src);
      ensure_node(dst);
      g.edges.push_back({src, dst, label});
      continue;
    }

    if (t.find("->") != std::string::npos) {
      throw std::runtime_error("Cannot parse edge line in " + path + ": " + t);
    }

    // Skip graph-level attributes like: rankdir=LR;
    if (t.find('=') != std::string::npos && t.find('[') == std::string::npos) {
      continue;
    }

    if (std::regex_match(t, m, node_re)) {
      const std::string node = strip_quotes(m[1].str());
      const std::string nu = to_upper(node);
      if (nu == "NODE" || nu == "EDGE" || nu == "GRAPH") {
        continue;
      }
      ensure_node(node);
    }
  }

  if (g.nodes.empty()) {
    throw std::runtime_error("No nodes parsed from DOT: " + path);
  }
  return g;
}

static CandidateGraphTensor candidate_from_dot(const std::string &path,
                                               const DatasetType dataset_type) {
  const DotGraph g = parse_dot_file(path);

  CandidateGraphTensor out;
  out.node_uid = g.nodes;

  std::unordered_map<std::string, int64_t> node_to_idx;
  node_to_idx.reserve(g.nodes.size());
  for (size_t i = 0; i < g.nodes.size(); ++i) {
    node_to_idx.emplace(g.nodes[i], static_cast<int64_t>(i));
  }

  if (dataset_type == DatasetType::BITMASK) {
    size_t bit_len = 0;
    for (const std::string &node : g.nodes) {
      if (bit_len == 0) {
        bit_len = node.size();
      }
      if (node.size() != bit_len) {
        throw std::runtime_error("Inconsistent bitmask length in DOT: " + path);
      }
      for (char ch : node) {
        if (ch == '0') {
          out.node_features.push_back(0.0F);
        } else if (ch == '1') {
          out.node_features.push_back(1.0F);
        } else {
          throw std::runtime_error("Non-binary BITMASK node '" + node +
                                   "' in DOT: " + path);
        }
      }
    }
    out.feat_dim = static_cast<int64_t>(bit_len);
  } else {
    out.feat_dim = 1;
    out.node_features.reserve(g.nodes.size());
    for (const std::string &node : g.nodes) {
      try {
        out.node_features.push_back(std::stof(node));
      } catch (...) {
        throw std::runtime_error(
            "Node '" + node +
            "' is not numeric (required for HASHED/MAPPED) in DOT: " + path);
      }
    }
  }

  out.edge_src_local.reserve(g.edges.size());
  out.edge_dst_local.reserve(g.edges.size());
  out.edge_attr.reserve(g.edges.size());
  for (const DotEdge &e : g.edges) {
    const auto it_src = node_to_idx.find(e.src);
    const auto it_dst = node_to_idx.find(e.dst);
    if (it_src == node_to_idx.end() || it_dst == node_to_idx.end()) {
      throw std::runtime_error("Internal parse mismatch while encoding edges.");
    }
    out.edge_src_local.push_back(it_src->second);
    out.edge_dst_local.push_back(it_dst->second);
    out.edge_attr.push_back(e.label);
  }

  return out;
}

static int64_t num_nodes(const CandidateGraphTensor &c) {
  if (c.feat_dim <= 0) {
    throw std::runtime_error("feat_dim must be > 0.");
  }
  if (c.node_features.size() % static_cast<size_t>(c.feat_dim) != 0U) {
    throw std::runtime_error("node_features size is not divisible by feat_dim.");
  }
  return static_cast<int64_t>(c.node_features.size() /
                              static_cast<size_t>(c.feat_dim));
}

static void validate_candidate(const CandidateGraphTensor &c) {
  const int64_t n_nodes = num_nodes(c);
  if (c.edge_src_local.size() != c.edge_dst_local.size() ||
      c.edge_src_local.size() != c.edge_attr.size()) {
    throw std::runtime_error(
        "edge_src_local, edge_dst_local, edge_attr must have same size.");
  }
  if (!c.node_uid.empty() && static_cast<int64_t>(c.node_uid.size()) != n_nodes) {
    throw std::runtime_error(
        "node_uid must be empty or exactly one UID per node.");
  }
  for (size_t i = 0; i < c.edge_src_local.size(); ++i) {
    const int64_t src = c.edge_src_local[i];
    const int64_t dst = c.edge_dst_local[i];
    if (src < 0 || dst < 0 || src >= n_nodes || dst >= n_nodes) {
      throw std::runtime_error("Local edge endpoint out of range.");
    }
  }
}

static void append_node_row(std::vector<float> &dst,
                            const CandidateGraphTensor &src,
                            const int64_t local_idx) {
  const size_t begin = static_cast<size_t>(local_idx * src.feat_dim);
  const size_t end = begin + static_cast<size_t>(src.feat_dim);
  dst.insert(dst.end(), src.node_features.begin() + static_cast<std::ptrdiff_t>(begin),
             src.node_features.begin() + static_cast<std::ptrdiff_t>(end));
}

static bool row_equals(const std::vector<float> &lhs, const int64_t lhs_feat,
                       const int64_t lhs_row, const std::vector<float> &rhs,
                       const int64_t rhs_feat, const int64_t rhs_row) {
  if (lhs_feat != rhs_feat) return false;
  const size_t l0 = static_cast<size_t>(lhs_row * lhs_feat);
  const size_t r0 = static_cast<size_t>(rhs_row * rhs_feat);
  for (int64_t j = 0; j < lhs_feat; ++j) {
    if (lhs[l0 + static_cast<size_t>(j)] != rhs[r0 + static_cast<size_t>(j)]) {
      return false;
    }
  }
  return true;
}

struct EdgeKey {
  int64_t src;
  int64_t dst;
  int64_t label;
  bool operator==(const EdgeKey &o) const noexcept {
    return src == o.src && dst == o.dst && label == o.label;
  }
};

struct EdgeKeyHash {
  size_t operator()(const EdgeKey &k) const noexcept {
    size_t h = std::hash<int64_t>{}(k.src);
    h ^= std::hash<int64_t>{}(k.dst) + 0x9e3779b97f4a7c15ULL + (h << 6U) +
         (h >> 2U);
    h ^= std::hash<int64_t>{}(k.label) + 0x9e3779b97f4a7c15ULL + (h << 6U) +
         (h >> 2U);
    return h;
  }
};

static CombinedFrontierGraphTensor
combine_frontier_separated(const std::vector<CandidateGraphTensor> &candidates) {
  if (candidates.empty()) {
    throw std::runtime_error("Cannot combine an empty frontier.");
  }

  CombinedFrontierGraphTensor out;
  out.feat_dim = candidates.front().feat_dim;

  int64_t node_offset = 0;
  for (size_t cidx = 0; cidx < candidates.size(); ++cidx) {
    const CandidateGraphTensor &c = candidates[cidx];
    validate_candidate(c);
    if (c.feat_dim != out.feat_dim) {
      throw std::runtime_error("All candidates must share feat_dim.");
    }

    const int64_t n = num_nodes(c);
    out.node_features.insert(out.node_features.end(), c.node_features.begin(),
                             c.node_features.end());
    out.membership.insert(out.membership.end(), static_cast<size_t>(n),
                          static_cast<int64_t>(cidx));

    for (size_t e = 0; e < c.edge_src_local.size(); ++e) {
      out.edge_src.push_back(c.edge_src_local[e] + node_offset);
      out.edge_dst.push_back(c.edge_dst_local[e] + node_offset);
      out.edge_attr.push_back(c.edge_attr[e]);
    }
    node_offset += n;
  }

  return out;
}

static CombinedFrontierGraphTensor
combine_frontier_merged(const std::vector<CandidateGraphTensor> &candidates) {
  if (candidates.empty()) {
    throw std::runtime_error("Cannot combine an empty frontier.");
  }

  CombinedFrontierGraphTensor out;
  out.feat_dim = candidates.front().feat_dim;

  std::unordered_map<std::string, int64_t> uid_to_global;
  std::unordered_set<EdgeKey, EdgeKeyHash> seen_edges;

  for (size_t cidx = 0; cidx < candidates.size(); ++cidx) {
    const CandidateGraphTensor &c = candidates[cidx];
    validate_candidate(c);

    if (c.feat_dim != out.feat_dim) {
      throw std::runtime_error("All candidates must share feat_dim.");
    }
    if (c.node_uid.empty()) {
      throw std::runtime_error("Merged mode requires node_uid for each node.");
    }

    const int64_t n = num_nodes(c);
    std::vector<int64_t> local_to_global(static_cast<size_t>(n), -1);

    for (int64_t local = 0; local < n; ++local) {
      const std::string &uid = c.node_uid[static_cast<size_t>(local)];
      int64_t global = -1;

      const auto it = uid_to_global.find(uid);
      if (it == uid_to_global.end()) {
        global = static_cast<int64_t>(out.membership.size());
        uid_to_global.emplace(uid, global);
        append_node_row(out.node_features, c, local);
        out.membership.push_back(static_cast<int64_t>(cidx));
      } else {
        global = it->second;
        if (!row_equals(out.node_features, out.feat_dim, global, c.node_features,
                        c.feat_dim, local)) {
          throw std::runtime_error(
              "Same node_uid has inconsistent node features: " + uid);
        }
      }

      local_to_global[static_cast<size_t>(local)] = global;
      out.pool_node_index.push_back(global);
      out.pool_membership.push_back(static_cast<int64_t>(cidx));
    }

    for (size_t e = 0; e < c.edge_src_local.size(); ++e) {
      const int64_t src = local_to_global[static_cast<size_t>(c.edge_src_local[e])];
      const int64_t dst = local_to_global[static_cast<size_t>(c.edge_dst_local[e])];
      const int64_t lbl = c.edge_attr[e];

      const EdgeKey key{src, dst, lbl};
      if (seen_edges.insert(key).second) {
        out.edge_src.push_back(src);
        out.edge_dst.push_back(dst);
        out.edge_attr.push_back(lbl);
      }
    }
  }

  return out;
}

static CombinedFrontierGraphTensor
combine_frontier(const std::vector<CandidateGraphTensor> &candidates,
                 const CombineMode mode) {
  return (mode == CombineMode::Merged) ? combine_frontier_merged(candidates)
                                       : combine_frontier_separated(candidates);
}

static GoalGraphTensor goal_from_candidate(const CandidateGraphTensor &c) {
  GoalGraphTensor g;
  g.feat_dim = c.feat_dim;
  g.node_features = c.node_features;
  g.edge_src = c.edge_src_local;
  g.edge_dst = c.edge_dst_local;
  g.edge_attr = c.edge_attr;
  return g;
}

static std::vector<int64_t> make_edge_index_2xE(const std::vector<int64_t> &src,
                                                const std::vector<int64_t> &dst) {
  if (src.size() != dst.size()) {
    throw std::runtime_error("edge src/dst size mismatch.");
  }
  std::vector<int64_t> out(2U * src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = src[i];
    out[src.size() + i] = dst[i];
  }
  return out;
}

class OnnxFrontierRunner {
public:
  explicit OnnxFrontierRunner(const std::string &onnx_path)
      : env_(ORT_LOGGING_LEVEL_WARNING, "RLFrontierInfer"),
        memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), session_options_);
    input_names_ = session_->GetInputNames();
    output_names_ = session_->GetOutputNames();
    expects_goal_branch_ = has_input("goal_node_features");
    mask_length_hint_ = infer_mask_length_hint();
  }

  bool expects_goal_branch() const { return expects_goal_branch_; }

  std::vector<float> run(const CombinedFrontierGraphTensor &frontier,
                         const size_t n_candidates,
                         const std::optional<GoalGraphTensor> &goal,
                         const std::optional<size_t> &mask_len_override) const {
    if (n_candidates == 0U) {
      throw std::runtime_error("Empty frontier: no candidates.");
    }

    const size_t n_nodes = frontier.membership.size();
    const size_t feat_dim = static_cast<size_t>(frontier.feat_dim);
    if (feat_dim == 0U) {
      throw std::runtime_error("frontier.feat_dim must be > 0.");
    }
    if (frontier.node_features.size() != n_nodes * feat_dim) {
      throw std::runtime_error("Invalid node_features size for [N,F].");
    }
    if (frontier.edge_src.size() != frontier.edge_dst.size() ||
        frontier.edge_src.size() != frontier.edge_attr.size()) {
      throw std::runtime_error("Invalid edge vectors size mismatch.");
    }

    std::vector<float> node_features = frontier.node_features;
    std::vector<int64_t> membership = frontier.membership;
    std::vector<int64_t> edge_attr = frontier.edge_attr;
    std::vector<int64_t> edge_index_data =
        make_edge_index_2xE(frontier.edge_src, frontier.edge_dst);

    std::vector<float> goal_node_features;
    std::vector<int64_t> goal_edge_attr;
    std::vector<int64_t> goal_edge_index_data;
    std::vector<int64_t> goal_batch;
    size_t gn = 0U;
    size_t ge = 0U;
    size_t goal_feat_dim = 0U;

    if (goal.has_value()) {
      goal_feat_dim = static_cast<size_t>(goal->feat_dim);
      if (goal_feat_dim == 0U) {
        throw std::runtime_error("goal feat_dim must be > 0.");
      }
      if (goal->node_features.size() % goal_feat_dim != 0U) {
        throw std::runtime_error("Invalid goal node_features size.");
      }
      if (goal_feat_dim != feat_dim) {
        throw std::runtime_error(
            "Goal feat_dim and frontier feat_dim differ. They must match.");
      }
      if (goal->edge_src.size() != goal->edge_dst.size() ||
          goal->edge_src.size() != goal->edge_attr.size()) {
        throw std::runtime_error("Invalid goal edge vectors size mismatch.");
      }

      goal_node_features = goal->node_features;
      goal_edge_attr = goal->edge_attr;
      goal_edge_index_data = make_edge_index_2xE(goal->edge_src, goal->edge_dst);
      gn = goal_node_features.size() / goal_feat_dim;
      ge = goal_edge_attr.size();
      goal_batch.assign(gn, 0); // single frontier => one goal graph id
    }

    // Priority for mask length:
    // 1) explicit CLI override (--mask-len)
    // 2) static ONNX shape hint (if available)
    // 3) fallback (commonly traced to 32 in exported models)
    constexpr size_t kMaskFallbackLen = 32U;
    const size_t mask_len = mask_len_override.has_value()
                                ? *mask_len_override
                                : ((mask_length_hint_ > 0U) ? mask_length_hint_
                                                            : std::max(kMaskFallbackLen, n_candidates));
    if (n_candidates > mask_len) {
      throw std::runtime_error(
          "Frontier size exceeds model mask capacity (" +
          std::to_string(n_candidates) + " > " + std::to_string(mask_len) + ").");
    }

    std::unique_ptr<bool[]> mask(new bool[mask_len]);
    for (size_t i = 0; i < mask_len; ++i) mask[i] = (i < n_candidates);

    std::vector<int64_t> candidate_batch(n_candidates, 0);
    std::vector<int64_t> action_map(n_candidates, 0);
    std::iota(action_map.begin(), action_map.end(), 0);

    std::vector<int64_t> pool_node_index = frontier.pool_node_index;
    std::vector<int64_t> pool_membership = frontier.pool_membership;

    float dummy_f = 0.0F;
    int64_t dummy_i64 = 0;

    auto ptr_f = [&](std::vector<float> &v) -> float * {
      return v.empty() ? &dummy_f : v.data();
    };
    auto ptr_i64 = [&](std::vector<int64_t> &v) -> int64_t * {
      return v.empty() ? &dummy_i64 : v.data();
    };

    const size_t n_edges = edge_attr.size();
    const std::array<int64_t, 2> node_shape{
        static_cast<int64_t>(n_nodes), static_cast<int64_t>(feat_dim)};
    const std::array<int64_t, 2> edge_index_shape{
        2, static_cast<int64_t>(n_edges)};
    const std::array<int64_t, 2> edge_attr_shape{
        static_cast<int64_t>(n_edges), 1};
    const std::array<int64_t, 1> membership_shape{
        static_cast<int64_t>(n_nodes)};
    const std::array<int64_t, 1> mask_shape{static_cast<int64_t>(mask_len)};

    const std::array<int64_t, 2> goal_node_shape{
        static_cast<int64_t>(gn), static_cast<int64_t>(goal_feat_dim)};
    const std::array<int64_t, 2> goal_edge_index_shape{
        2, static_cast<int64_t>(ge)};
    const std::array<int64_t, 2> goal_edge_attr_shape{
        static_cast<int64_t>(ge), 1};
    const std::array<int64_t, 1> goal_batch_shape{
        static_cast<int64_t>(gn)};

    const std::array<int64_t, 1> candidate_batch_shape{
        static_cast<int64_t>(n_candidates)};
    const std::array<int64_t, 1> action_map_shape{
        static_cast<int64_t>(n_candidates)};
    const std::array<int64_t, 1> pool_shape{
        static_cast<int64_t>(pool_node_index.size())};

    std::vector<const char *> input_names_cstr;
    input_names_cstr.reserve(input_names_.size());
    for (const auto &s : input_names_) input_names_cstr.push_back(s.c_str());

    std::vector<const char *> output_names_cstr;
    output_names_cstr.reserve(output_names_.size());
    for (const auto &s : output_names_) output_names_cstr.push_back(s.c_str());

    std::vector<Ort::Value> inputs;
    inputs.reserve(input_names_.size());

    for (const std::string &name : input_names_) {
      if (name == "node_features") {
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info_, ptr_f(node_features), node_features.size(),
            node_shape.data(), node_shape.size()));
      } else if (name == "edge_index") {
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_, ptr_i64(edge_index_data), edge_index_data.size(),
            edge_index_shape.data(), edge_index_shape.size()));
      } else if (name == "edge_attr") {
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_, ptr_i64(edge_attr), edge_attr.size(),
            edge_attr_shape.data(), edge_attr_shape.size()));
      } else if (name == "membership") {
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_, ptr_i64(membership), membership.size(),
            membership_shape.data(), membership_shape.size()));
      } else if (name == "mask") {
        inputs.emplace_back(Ort::Value::CreateTensor<bool>(
            memory_info_, mask.get(), mask_len, mask_shape.data(),
            mask_shape.size()));
      } else if (name == "goal_node_features") {
        if (!goal.has_value()) {
          throw std::runtime_error(
              "Model requires goal_* inputs but no goal graph was provided.");
        }
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info_, ptr_f(goal_node_features), goal_node_features.size(),
            goal_node_shape.data(), goal_node_shape.size()));
      } else if (name == "goal_edge_index") {
        if (!goal.has_value()) {
          throw std::runtime_error(
              "Model requires goal_* inputs but no goal graph was provided.");
        }
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_, ptr_i64(goal_edge_index_data), goal_edge_index_data.size(),
            goal_edge_index_shape.data(), goal_edge_index_shape.size()));
      } else if (name == "goal_edge_attr") {
        if (!goal.has_value()) {
          throw std::runtime_error(
              "Model requires goal_* inputs but no goal graph was provided.");
        }
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_, ptr_i64(goal_edge_attr), goal_edge_attr.size(),
            goal_edge_attr_shape.data(), goal_edge_attr_shape.size()));
      } else if (name == "goal_batch") {
        if (!goal.has_value()) {
          throw std::runtime_error(
              "Model requires goal_* inputs but no goal graph was provided.");
        }
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_, ptr_i64(goal_batch), goal_batch.size(),
            goal_batch_shape.data(), goal_batch_shape.size()));
      } else if (name == "candidate_batch") {
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_, ptr_i64(candidate_batch), candidate_batch.size(),
            candidate_batch_shape.data(), candidate_batch_shape.size()));
      } else if (name == "action_map") {
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_, ptr_i64(action_map), action_map.size(),
            action_map_shape.data(), action_map_shape.size()));
      } else if (name == "pool_node_index") {
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_, ptr_i64(pool_node_index), pool_node_index.size(),
            pool_shape.data(), pool_shape.size()));
      } else if (name == "pool_membership") {
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_, ptr_i64(pool_membership), pool_membership.size(),
            pool_shape.data(), pool_shape.size()));
      } else {
        throw std::runtime_error("Unsupported model input name: " + name);
      }
    }

    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names_cstr.data(),
                                 inputs.data(), inputs.size(),
                                 output_names_cstr.data(), output_names_cstr.size());
    if (outputs.empty() || !outputs[0].IsTensor()) {
      throw std::runtime_error("ONNX model returned no tensor output.");
    }

    const auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
    const size_t out_len = out_info.GetElementCount();
    if (out_len < n_candidates) {
      throw std::runtime_error(
          "Output length < number of candidates: " + std::to_string(out_len) +
          " < " + std::to_string(n_candidates));
    }

    std::vector<float> logits(out_len, 0.0F);
    const ONNXTensorElementDataType elem = out_info.GetElementType();
    if (elem == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      const float *ptr = outputs[0].GetTensorData<float>();
      std::copy(ptr, ptr + static_cast<std::ptrdiff_t>(out_len), logits.begin());
    } else if (elem == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
      const double *ptr = outputs[0].GetTensorData<double>();
      for (size_t i = 0; i < out_len; ++i) logits[i] = static_cast<float>(ptr[i]);
    } else {
      throw std::runtime_error("Output tensor is not float/double.");
    }

    return logits;
  }

private:
  bool has_input(const std::string &name) const {
    return std::find(input_names_.begin(), input_names_.end(), name) !=
           input_names_.end();
  }

  size_t infer_mask_length_hint() const {
    if (!output_names_.empty()) {
      const auto out_info = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
      const auto shape = out_info.GetShape();
      if (!shape.empty() && shape[0] > 0) {
        return static_cast<size_t>(shape[0]);
      }
    }

    for (size_t i = 0; i < input_names_.size(); ++i) {
      if (input_names_[i] == "mask") {
        const auto in_info = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        const auto shape = in_info.GetShape();
        if (!shape.empty() && shape[0] > 0) {
          return static_cast<size_t>(shape[0]);
        }
      }
    }
    return 0U;
  }

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_info_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  bool expects_goal_branch_ = false;
  size_t mask_length_hint_ = 0;
};

struct Args {
  std::string onnx_path;
  CombineMode mode = CombineMode::Merged;
  DatasetType dataset_type = DatasetType::HASHED;
  std::vector<std::string> candidates;
  std::optional<std::string> goal_path;
  std::optional<size_t> mask_len;
};

static void print_usage(const char *argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0
      << " --onnx MODEL.onnx --mode merged|separated --dataset-type HASHED|MAPPED|BITMASK\n"
      << "    --candidate c1.dot --candidate c2.dot [--candidate ...]\n"
      << "    [--goal goal.dot] [--frontier-list frontier.txt] [--mask-len N]\n\n"
      << "Notes:\n"
      << "  - In separated mode, --goal is required.\n"
      << "  - --frontier-list is a text file with one candidate DOT path per line.\n"
      << "  - --mask-len forces the mask size (useful for traced ONNX models, e.g. 32).\n";
}

static Args parse_args(int argc, char **argv) {
  if (argc <= 1) {
    print_usage(argv[0]);
    throw std::runtime_error("No arguments provided.");
  }

  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const std::string &flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value after " + flag);
      }
      ++i;
      return argv[i];
    };

    if (a == "--help" || a == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (a == "--onnx") {
      args.onnx_path = need(a);
    } else if (a == "--mode") {
      args.mode = parse_mode(need(a));
    } else if (a == "--dataset-type") {
      args.dataset_type = parse_dataset_type(need(a));
    } else if (a == "--candidate") {
      args.candidates.push_back(need(a));
    } else if (a == "--frontier-list") {
      const auto lines = read_nonempty_lines(need(a));
      args.candidates.insert(args.candidates.end(), lines.begin(), lines.end());
    } else if (a == "--goal") {
      args.goal_path = need(a);
    } else if (a == "--mask-len") {
      const std::string val = need(a);
      size_t parsed = 0U;
      try {
        const unsigned long long tmp = std::stoull(val);
        if (tmp == 0ULL) {
          throw std::runtime_error("mask-len must be > 0");
        }
        parsed = static_cast<size_t>(tmp);
      } catch (const std::exception &) {
        throw std::runtime_error("Invalid --mask-len value: " + val +
                                 " (expected positive integer)");
      }
      args.mask_len = parsed;
    } else {
      throw std::runtime_error("Unknown argument: " + a);
    }
  }

  if (args.onnx_path.empty()) {
    throw std::runtime_error("Missing --onnx MODEL.onnx");
  }
  if (args.candidates.empty()) {
    throw std::runtime_error("No frontier candidates provided.");
  }
  if (args.mode == CombineMode::Separated && !args.goal_path.has_value()) {
    throw std::runtime_error("Separated mode requires --goal goal.dot");
  }

  return args;
}

int main(int argc, char **argv) {
  try {
    const Args args = parse_args(argc, argv);

    std::vector<CandidateGraphTensor> candidates;
    candidates.reserve(args.candidates.size());
    for (const std::string &p : args.candidates) {
      candidates.push_back(candidate_from_dot(p, args.dataset_type));
    }

    const CombinedFrontierGraphTensor combined = combine_frontier(candidates, args.mode);

    std::optional<GoalGraphTensor> goal;
    if (args.goal_path.has_value()) {
      goal = goal_from_candidate(candidate_from_dot(*args.goal_path, args.dataset_type));
    }

    OnnxFrontierRunner runner(args.onnx_path);
    if (runner.expects_goal_branch() && !goal.has_value()) {
      throw std::runtime_error(
          "This ONNX model expects goal_* inputs, but --goal was not provided.");
    }

    const std::vector<float> raw_output =
        runner.run(combined, args.candidates.size(), goal, args.mask_len);
    std::cout << "Model raw output (" << raw_output.size() << " values): [";
    for (size_t i = 0; i < raw_output.size(); ++i) {
      if (i != 0U) {
        std::cout << ", ";
      }
      std::cout << std::fixed << std::setprecision(6) << raw_output[i];
    }
    std::cout << "]\n";

    std::vector<float> logits(raw_output.begin(),
                              raw_output.begin() +
                                  static_cast<std::ptrdiff_t>(args.candidates.size()));

    std::vector<size_t> order(args.candidates.size());
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(),
                     [&](const size_t a, const size_t b) {
                       return logits[a] > logits[b];
                     });

    std::cout << "Frontier ranking (best first):\n";
    for (size_t rank = 0; rank < order.size(); ++rank) {
      const size_t idx = order[rank];
      std::cout << std::setw(3) << (rank + 1) << "."
                << " candidate_idx=" << idx
                << " logit=" << std::fixed << std::setprecision(6) << logits[idx]
                << " path=" << args.candidates[idx] << "\n";
    }

    std::cout << "\nBest candidate index: " << order.front()
              << "\nBest candidate path : " << args.candidates[order.front()]
              << "\n";
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
