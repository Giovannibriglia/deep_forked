#pragma once
#include "KripkeState.h"
#include "State.h"
#include "neuralnets/FringeEvalRL.h"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>

#include "GraphNN.h"

/**
 * \struct FringeTensor
 * \brief Represents a Fringe in tensor format for input to the RL
 * evaluator using ONNX.
 *
 * This structure encapsulates the graph as a set of arrays:
 * - edge_src: 1D array of symbolic source node IDs for each edge.
 * - edge_dst: 1D array of symbolic destination node IDs for each edge.
 * - edge_attrs: 1D array of edge attributes or labels, aligned with edges.
 * - real_node_ids: 1D array mapping symbolic node IDs to their corresponding
 * - real_node_ids_bitmask: flatten multiDim array mapping symbolic node IDs to
 * their corresponding BITMASK IDs.
 *
 */

struct FringeTensor {
  std::vector<uint64_t> edge_src;
  ///< [1, num_edges] -- First dimension.
  ///< Symbolic source node ID for each edge.
  std::vector<uint64_t>
      edge_dst; ///< [1, num_edges] -- Second dimension. Symbolic destination
                ///< node ID for each edge.

  /// edge_src and edge_dest are used to create edge_index -> list <edge_source,
  /// edge_target> -> [2, num_edges]

  std::vector<uint64_t>
      edge_attrs; ///< [1, num_edges] Edge attributes or labels,

  ///< aligned with edge_ids.
  std::vector<uint64_t> real_node_ids;
  ///< [num_nodes, 1] Mapping from symbolic
  ///< node IDs to real/hashed node IDs.
  ///< aligned with edge_ids.

  std::vector<uint64_t> membership;
  ///< [num_states, 1] mapping each state to the corresponding start of the
  ///< array in real_nodes_ids.

  std::vector<uint8_t> active_states = std::vector<uint8_t>(
      ArgumentParser::get_instance().get_RL_fringe_size(), 0);
  ///< [32, 1] boolean masks that it is set to 1 for each active state.

  std::vector<int64_t> candidate_batch;
  ///< A zero for each State (kstate and goal nodes) for internal operations.

  std::vector<uint8_t> real_node_ids_bitmask;
  ///< Special Case: BITMASK nodes have BITMASKS as real IDs (lists of 0-1)
  ///< flattened in a single vector (use uint for easier conversion)
};

/**
 * \class FringeEvalRL
 * \brief Singleton class for RL-based evaluation.
 *
 * This class provides an interface for evaluating a fringe of states using RL.
 * It is implemented as a singleton, ensuring only one
 * instance exists during the application's lifetime.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date April 9, 2026
 */
template <StateRepresentation StateRepr> class FringeEvalRL {
public:
  /**
   * \brief Get the singleton instance of FringeEvalRL.
   * \return Reference to the singleton instance.
   */
  static FringeEvalRL &get_instance();

  /**
   * \brief Create the singleton instance of FringeEvalRL.
   */

  static void create_instance();

  /**
   * \brief Get the scores for a set of states (fringe) using RL
   * using native C++ code \tparam StateRepr The state representation type.
   * \param states The set of states to evaluate.
   * \return The relative score for the states in the fringe.
   */
  [[nodiscard]] std::vector<float>
  get_score(const std::vector<State<StateRepr>> &states);

  /** \brief Deleted copy constructor (singleton pattern). */
  FringeEvalRL(const FringeEvalRL &) = delete;

  /** \brief Deleted copy assignment operator (singleton pattern). */
  FringeEvalRL &operator=(const FringeEvalRL &) = delete;

  /** \brief Deleted move constructor (singleton pattern). */
  FringeEvalRL(FringeEvalRL &&) = delete;

  /** \brief Deleted move assignment operator (singleton pattern). */
  FringeEvalRL &operator=(FringeEvalRL &&) = delete;

private:
  /**
   * \brief Private constructor for singleton pattern.
   */
  FringeEvalRL();

  static FringeEvalRL *instance; ///< Singleton instance pointer

  std::string m_model_path =
      Configuration::get_instance()
          .get_GNN_model_path(); ///< Path to the RL model

  ///// --- ONNX Runtime inference components ---
  Ort::Env m_env{
      ORT_LOGGING_LEVEL_ERROR,
      "FringeEvalRLEnv"}; ///< ONNX Runtime environment for Fringe inference.
  Ort::SessionOptions m_session_options; ///< ONNX Runtime session options.
  std::unique_ptr<Ort::Session>
      m_session; ///< Pointer to the ONNX Runtime session.
  std::unique_ptr<Ort::AllocatorWithDefaultOptions>
      m_allocator; ///< Allocator for ONNX Runtime memory management.
  std::unique_ptr<Ort::MemoryInfo>
      m_memory_info; ///< Memory info for ONNX Runtime tensors.

  std::vector<std::string>
      m_input_names; ///< Names of the input nodes for the ONNX model.
  std::vector<std::string>
      m_output_names; ///< Names of the output nodes for the ONNX model.

  bool m_model_loaded =
      false; ///< Indicates whether the ONNX model has been loaded.

  /**
   * \brief Converts a set of KripkeState (Fringe) to a minimal GraphTensor
   * representation.
   *
   * This function transforms the given Fringe into a FringeTensor,
   * extracting only the essential information required for GNN input.
   *
   * \param states The set of States to convert.
   * \return A FringeTensor containing the minimal tensor representation of the
   * graph.
   */
  [[nodiscard]] FringeTensor
  fringe_to_tensor_minimal(const std::vector<State<StateRepr>> &states);

  /**
   * \brief Initializes the ONNX Runtime model for GNN inference.
   *
   * Sets up the ONNX Runtime environment, session options, loads the GNN model,
   * and prepares input/output names and memory information required for
   * inference. This function should be called before performing any inference
   * with the model.
   */
  void initialize_onnx_model();

  /**
   * \brief Runs ONNX Runtime inference on the provided GraphTensor.
   *
   * This method takes a GraphTensor representing the input graph, prepares the
   * necessary ONNX Runtime tensors, and performs inference using the loaded GNN
   * model. It returns the resulting score or output from the neural network.
   *
   * \param tensor The GraphTensor containing the graph data for inference.
   * \return The output score from the GNN model as a float.
   */
  float run_inference(const GraphTensor &tensor) const;

  /** \brief Function that return the position of each state in the fringe wrt
   * to their score
   *
   * \param scores The Array of scores ordered as in input
   * \return THe array that associates to each position the rank of the states
   */
  std::vector<float> rankScores(const std::vector<float> &scores) const;
};

#include "FringeEvalRL.tpp"
