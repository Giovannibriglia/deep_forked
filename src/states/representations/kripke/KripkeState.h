/**
 * \class KripkeState
 * \brief Class that represents a Kripke Structure for epistemic planning.
 *
 * \details  A Kripke Structure is the standard way of representing e-States in
 * Epistemic Planning. See KripkeWorld and KripkeStorage for related structures.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano.
 * \date May 17, 2025
 */
#pragma once

#include "KripkeWorld.h"
#include "actions/Action.h"
#include "bisimulation/Bisimulation.h"
#include "neuralnets/GraphTensor.h"
#include "utilities/Define.h"

class KripkeState {
public:
  // --- Constructors/Destructor ---
  KripkeState() = default;

  /**
   * \brief Copy constructor.
   * \param other The KripkeState to copy from.
   */
  KripkeState(const KripkeState &other);

  ~KripkeState() = default;

  // --- Setters ---
  /** \brief Set the set of worlds for this KripkeState.
   *  \param[in] to_set The set of KripkeWorld pointers to assign.
   */
  void set_worlds(const KripkeWorldPointersSet &to_set);

  /** \brief Set the pointed world for this KripkeState.
   *
   *  Also resets the designated-worlds set to `{to_set}` so the multi-pointed
   *  invariant (`m_pointed ∈ m_designated_worlds`, size == 1 in the
   *  single-pointed default) holds. To install a true multi-pointed
   *  designated set, call \ref set_designated_worlds instead.
   *  \param[in] to_set The KripkeWorld pointer to assign as pointed.
   */
  void set_pointed(const KripkeWorldPointer &to_set);

  /** \brief Install a (possibly multi-pointed) designated-worlds set.
   *
   *  Sets `m_designated_worlds = to_set` and picks `m_pointed` as the
   *  canonical element (the smallest by `KripkeWorldPointer::operator<`).
   *  Asserts that `to_set` is non-empty.
   *  \param[in] to_set The designated worlds.
   */
  void set_designated_worlds(const KripkeWorldPointersSet &to_set);

  /** \brief Set the beliefs map for this KripkeState.
   *  \param[in] to_set The beliefs map to assign.
   */
  void set_beliefs(const KripkeWorldPointersTransitiveMap &to_set);

  /** \brief Clears the beliefs map for this KripkeState.
   * This is only usable by Bisimulation.
   */
  void clear_beliefs();

  /** \brief Set the maximum depth for this KripkeState.
   *  \param[in] to_set The unsigned int value to assign as max depth.
   */
  void set_max_depth(unsigned int to_set) noexcept;

  // --- Getters ---
  /** \brief Get the set of worlds in this KripkeState.
   *  \return The set of KripkeWorld pointers.
   */
  [[nodiscard]] const KripkeWorldPointersSet &get_worlds() const noexcept;

  /** \brief Get the vector of worlds in this KripkeState.
   *  \return The vector of KripkeWorld pointers.
   */
  [[nodiscard]] const KripkeWorldPointersVec &get_worlds_vec() const noexcept;

  /** \brief Get the canonical pointed world in this KripkeState.
   *
   *  For single-pointed states this is the unique designated world; for
   *  multi-pointed states it is the canonical pick (smallest by
   *  `KripkeWorldPointer::operator<`) of \ref m_designated_worlds. Callers
   *  that need the full set should use \ref get_designated_worlds.
   *  \return The canonical pointed KripkeWorld pointer.
   */
  [[nodiscard]] const KripkeWorldPointer &get_pointed() const noexcept;

  /** \brief Get the designated-worlds set.
   *
   *  Always contains \ref m_pointed. Size > 1 only after a DEL transition
   *  whose action had multiple designated events applicable at the input
   *  pointed (multi-pointed result).
   *  \return The designated-worlds set.
   */
  [[nodiscard]] const KripkeWorldPointersSet &
  get_designated_worlds() const noexcept;

  /** \brief True if more than one designated world (i.e., multi-pointed). */
  [[nodiscard]] bool is_multi_pointed() const noexcept;

  /** \brief Get the beliefs map in this KripkeState.
   *  \return The beliefs map.
   */
  [[nodiscard]] const KripkeWorldPointersTransitiveMap &
  get_beliefs() const noexcept;

  /** \brief Get the beliefs map in this KripkeState in vectorized form.
   *  \return The beliefs map.
   */
  [[nodiscard]] const KripkeWorldPointersTransitiveMapVec &
  get_beliefs_vec() const noexcept;

  /** \brief Get the maximum depth of this KripkeState.
   *  \return The max depth value.
   */
  [[nodiscard]] unsigned int get_max_depth() const noexcept;

  /** \brief Compute the successor state after applying an action.
   *  \param[in] act The action to apply.
   *  \return The resulting KripkeState.
   */
  [[nodiscard]] KripkeState compute_successor(const Action &act) const;

#ifndef USE_MASTAR
  /** \brief Full-DEL applicability check used by State::is_executable when a
   *  plank action is attached. Returns true iff some designated event of
   *  `act.get_del_action()` has its precondition entailed at some designated
   *  world of *this*.
   */
  [[nodiscard]] bool is_executable_del(const Action &act) const;
#endif

  // --- Operators ---
  /** \brief Copy Assignment operator.*/
  KripkeState &operator=(const KripkeState &to_copy);

  /** \brief Less-than operator for set operations.
   *  \param[in] to_compare The KripkeState to compare.
   *  \return True if this is less than to_compare, false otherwise.
   */
  [[nodiscard]] bool operator<(const KripkeState &to_compare) const;

  /** \brief Equal operator.
   *  \param[in] to_compare The KripkeState to compare.
   *  \return True if this is less than to_compare, false otherwise.
   */
  bool operator==(const KripkeState &to_compare) const;

  /// \name Needed for State<T>
  ///@{
  /** \brief Build the initial Kripke structure (choose method internally).
   */
  void build_initial();

  /** \brief Function that checks if *this* entails a Fluent.
   *
   *
   * @param to_check: the Fluent to check if is entailed by *this*.
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.
   */
  [[nodiscard]] bool entails(const Fluent &to_check) const;

  /** \brief Function that checks if *this* entails a conjunctive set of Fluent.
   *
   *
   * @param to_check: the conjunctive set of \ref Fluent to check if is entailed
   * by *this*.
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.*/
  [[nodiscard]] bool entails(const FluentsSet &to_check) const;

  /** \brief Function that checks if *this* entails a DNF \ref FluentFormula.
   *
   *
   * @param to_check: the DNF \ref FluentFormula to check if is entailed by
   * *this*.
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.*/
  [[nodiscard]] bool entails(const FluentFormula &to_check) const;

  /** \brief Function that checks if *this* entails a \ref BeliefFormula.
   *
   *
   * @param to_check: the \ref BeliefFormula to check if is entailed by *this*.
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.*/
  [[nodiscard]] bool entails(const BeliefFormula &to_check) const;

  /** \brief Function that checks if *this* entails a CNF \ref FormulaeList.
   *
   *
   *
   * @param to_check: the CNF \ref FormulaeList to check if is entailed by
   * *this*.
   *
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.*/
  [[nodiscard]] bool entails(const FormulaeList &to_check) const;

  /** \brief Function that applies bisimulation contraction to this*/
  void contract_with_bisimulation();

  [[nodiscard]] const GraphTensor &get_tensor_representation();

  /**
   * @brief Compacts the repetition indices of all worlds in ascending numeric
   * order.
   *
   * This method assigns new indices based on sorted order of the original IDs.
   * The result is a contiguous sequence starting from 0, ordered by numeric
   * rank.
   *
   * Example:
   *   Input  : {10, 3, 10, 7}
   *   Output : {2, 0, 2, 1}
   *
   * @return A new KripkeState with compacted repetition indices.
   */
  void compact_repetitions();

  ///}
  // --- Printing ---
  /** \brief Print this KripkeState.*/
  void print() const;

  /** \brief Print this KripkeState to a dot format in the file stream ofs.
   * Params: [in] ofs — The output file stream.*/
  void print_dot_format(std::ofstream &ofs) const;

  /** \brief Function that prints the information of *this* for the generation
   * of the dataset used to train the GNN. \param ofs The output stream to print
   * to.
   * if each dataset entry is merged <goal,state> or not.
   */
  void print_dataset_format(std::ofstream &ofs) const;

private:
  // --- Data members ---
  /** \brief Used to differentiate partial KripkeStates generated by partial
   * observation. */
  unsigned int m_max_depth = 0;
  /** \brief Set of pointers to each world in the structure. */
  KripkeWorldPointersSet m_worlds;
  /** \brief Canonical pointed world. Always equals `*m_designated_worlds.begin()`.
   *  Kept as a separate field so existing single-pointed code paths (mA* engine,
   *  printing, heuristics) can read it directly without traversing the set. */
  KripkeWorldPointer m_pointed;
  /** \brief Designated-worlds set. Invariant: non-empty after `set_pointed` /
   *  `set_designated_worlds`, and always contains `m_pointed`. Size 1 in
   *  single-pointed mode (default). Multi-pointed only after a DEL transition
   *  with more than one applicable designated event. */
  KripkeWorldPointersSet m_designated_worlds;
  /** \brief Beliefs of each agent in every world. */
  KripkeWorldPointersTransitiveMap m_beliefs;

  /** \brief Set of pointers to each world in the structure -- empty otherwise.
   */
  KripkeWorldPointersVec m_worlds_vec;
  /** \brief Beliefs of each agent in every world in vector form for strong
   * equivalence check -- empty otherwise. */
  KripkeWorldPointersTransitiveMapVec m_beliefs_vec;

  void set_worlds_vec();

  void set_beliefs_vec();

  /** \brief Tensor version of this for the various NN-based heuristics */
  GraphTensor m_tensor_representation;
  /** Guard to indicate whether the GraphTensor needs to be computed*/
  bool m_computed_tensor_representation = false;

  // --- Internal helpers ---
  /** \brief Add a world to the Kripke structure.
   *  \param[in] to_add The KripkeWorld to add.
   */
  void add_world(const KripkeWorld &to_add);

  /** \brief Add a belief edge for an agent between two worlds.
   *  \param[in] from The source world.
   *  \param[in] to The target world.
   *  \param[in] ag The agent.
   */
  void add_edge(const KripkeWorldPointer &from, const KripkeWorldPointer &to,
                const Agent &ag);

  /** \brief Add a world with repetition tracking.
   *  \param[in] to_add The KripkeWorld to add.
   *  \return Pointer to the newly inserted KripkeWorld.
   */
  KripkeWorldPointer add_rep_world(const KripkeWorld &to_add);

  /** \brief Add a world with old repetition tracking.
   *  \param[in] to_add The KripkeWorld to add.
   *  \param[in] old_repetition Used to distinguish from same level but
   * different origins. \return Pointer to the newly inserted KripkeWorld.
   */
  KripkeWorldPointer add_rep_world(const KripkeWorld &to_add,
                                   unsigned short old_repetition);

  /** \brief Add a world with repetition and newness tracking.
   *  \param[in] to_add The KripkeWorld to add.
   *  \param[in] repetition Used to distinguish from other levels.
   *  \param[out] is_new Indicates if the world was already present.
   *  \return Pointer to the newly inserted KripkeWorld.
   */
  KripkeWorldPointer add_rep_world(const KripkeWorld &to_add,
                                   unsigned short repetition, bool &is_new);

  // --- Structure Building ---

  /** \brief Generate all possible permutations of the domain's fluents.
   *  \param[out] permutation The permutation in construction.
   *  \param[in] index The index of the fluent to add.
   *  \param[in] initially_known The set of initially known fluents.
   */
  void generate_initial_worlds(FluentsSet &permutation, unsigned int index,
                               const FluentsSet &initially_known);

  /** \brief Check if a KripkeWorld respects initial conditions and add it if
   * so. \param[in] possible_add The KripkeWorld to check.
   */
  void add_initial_world(const KripkeWorld &possible_add);

  /** \brief Generate all initial edges for the KripkeState. */
  void generate_initial_edges();

  /** \brief Remove an edge for an agent between two worlds.
   *  \param[in] from The KripkeWorld pointer to remove the edge from.
   *  \param[in] to The KripkeWorld to remove.
   *  \param[in] ag The agent.
   */
  void remove_edge(const KripkeWorldPointer &from, const KripkeWorldPointer &to,
                   const Agent &ag);

  /** \brief Remove initial edges based on known fluent formula for an agent.
   *  \param[in] known_ff The fluent formula known by the agent.
   *  \param[in] ag The agent.
   */
  void remove_initial_edge(const FluentFormula &known_ff, const Agent &ag);

  /** \brief Remove initial edges based on a BeliefFormula.
   *  \param[in] to_check The BeliefFormula to check.
   */
  void remove_initial_edge_bf(const BeliefFormula &to_check);

  /** \brief Add beliefs to a world.
   *  \param[in] world The KripkeWorld pointer.
   *  \param[in] beliefs The beliefs map.
   */
  void add_world_beliefs(const KripkeWorldPointer &world,
                         const KripkeWorldPointersMap &beliefs);

  /** \brief Copy worlds and beliefs of oblivious agents to another KripkeState.
   *  \param[in] ret The new KripkeState.
   *  \param[in] oblivious_obs_agents The set of oblivious agents.
   */
  void maintain_oblivious_believed_worlds(
      KripkeState &ret, const AgentsSet &oblivious_obs_agents) const;

  // --- Transition/Execution ---
  /** \brief Recursively compute the result of an ontic action.
   *  \param[in] act The ontic action to apply.
   *  \param[in] ret The resulting KripkeState.
   *  \param[in] current_pw The world being currently calculated.
   *  \param[in] calculated Map tracking results of the transition function.
   *  \param[in] oblivious_obs_agents The set of oblivious agents.
   *  \return The resulting KripkeWorld pointer.
   */
  KripkeWorldPointer execute_ontic_helper(
      const Action &act, KripkeState &ret, const KripkeWorldPointer &current_pw,
      TransitionMap &calculated, AgentsSet &oblivious_obs_agents) const;

  /** \brief Recursively compute the result of a sensing/announcement action.
   *  \param[in] effects The effects of the action.
   *  \param[in] ret The resulting KripkeState.
   *  \param[in] current_pw The world being currently calculated.
   *  \param[in] calculated Map tracking results of the transition function.
   *  \param[in] partially_obs_agents The set of partially observant agents.
   *  \param[in] oblivious_obs_agents The set of oblivious agents.
   *  \param[in] previous_entailment The value of the coming state entailment.
   *  \return The resulting KripkeWorld pointer.
   */
  KripkeWorldPointer execute_sensing_announcement_helper(
      const FluentFormula &effects, KripkeState &ret,
      const KripkeWorldPointer &current_pw, TransitionMap &calculated,
      AgentsSet &partially_obs_agents, AgentsSet &oblivious_obs_agents,
      bool previous_entailment) const;

  /** \brief Apply an ontic action to this KripkeState.
   *  \param[in] act The ontic action to apply.
   *  \return The resulting KripkeState.
   */
  [[nodiscard]] KripkeState execute_ontic(const Action &act) const;

  /** \brief Apply a sensing action to this KripkeState.
   *  \param[in] act The sensing action to apply.
   *  \return The resulting KripkeState.
   */
  [[nodiscard]] KripkeState execute_sensing(const Action &act) const;

  /** \brief Apply an announcement action to this KripkeState.
   *  \param[in] act The announcement action to apply.
   *  \return The resulting KripkeState.
   */
  [[nodiscard]] KripkeState execute_announcement(const Action &act) const;

#ifndef USE_MASTAR
  /** \brief Full-DEL product update with a plank-grounded action carried by
   *  `act.get_del_action()`. Implementation lives in KripkeState.cpp under
   *  the same `!USE_MASTAR` guard.
   */
  [[nodiscard]] KripkeState compute_successor_del(const Action &act) const;

  /** \brief Populate worlds/edges/designated directly from plank's already-
   *  grounded del::state. Used when the (:init …) was given as an explicit
   *  Kripke structure (`:worlds … :relations … :labels … :designated …`),
   *  bypassing deep's S5-permutation enumeration. */
  void build_initial_from_plank_state();
#endif

  /*This is to allow bisimulation to reduce the size of the object*/
  friend class Bisimulation;
};
