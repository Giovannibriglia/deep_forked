#pragma once

#include <set>
#include <string>
#include <vector>

#include "formulae/BeliefFormula.h"
#include "utilities/Define.h"

namespace plank::del {
class action;
}

/**
 * \enum PropositionType
 * \brief Semantic category of an Action's effects/observability.
 *
 * Originally split between the parser's Proposition class and Action; now that
 * the EPDDL parser populates Action directly, the enum lives with Action.
 * EXECUTABILITY/OBSERVANCE/AWARENESS remain only as legacy NOTSET signals from
 * the old parser path and are no longer used as Action::m_type values.
 */
enum class PropositionType {
  EXECUTABILITY,
  ONTIC,
  SENSING,
  ANNOUNCEMENT,
  OBSERVANCE,
  AWARENESS,
  NOTSET
};

/**
 * \class Action
 * \brief Stores an action and all its information.
 * \author Francesco Fabiano
 * \date May 16, 2025
 * \copyright GNU Public License.
 */
class Action {
public:
  /// \name Constructors
  ///@{
  /** \brief Default constructor. */
  Action() = default;

  /**
   * \brief Constructor with a given name and id.
   * \param[in] name The value to assign to \ref m_name.
   * \param[in] id The value to assign to \ref m_id.
   */
  Action(const std::string &name, ActionId id);

  /**
   * \brief Copy constructor.
   * \param other The Action to copy from.
   */
  Action(const Action &) = default;
  Action(Action &&) noexcept = default;
  Action &operator=(const Action &) = default;
  Action &operator=(Action &&) noexcept = default;
  ~Action() = default;
  ///@}

  /// \name Getters and Setters
  ///@{
  /** \brief Gets the name of this action. */
  [[nodiscard]] std::string get_name() const;

  /** \brief Sets the name of this action.
   *  \param[in] name The value to assign to \ref m_name.
   */
  void set_name(const std::string &name);

  /** \brief Gets the executor agent of this action. */
  [[nodiscard]] Agent get_executor() const;

  /** \brief Sets the executor agent of this action.
   *  \param[in] executor The value to assign to \ref m_executor.
   */
  void set_executor(const Agent &executor);

  /** \brief Gets the unique id of this action. */
  [[nodiscard]] ActionId get_id() const;

  /** \brief Sets the unique id of this action.
   *  \param[in] id The value to assign to \ref m_id.
   */
  void set_id(ActionId id);

  /** \brief Gets the proposition type of this action. */
  [[nodiscard]] PropositionType get_type() const;

  /** \brief Sets the proposition type of this action.
   *  \param[in] type The value to assign to \ref m_type.
   */
  void set_type(PropositionType type);

  /** \brief Gets the executability conditions of this action. */
  [[nodiscard]] const FormulaeList &get_executability() const;

  /** \brief Gets the effects of this action. */
  [[nodiscard]] const EffectsMap &get_effects() const;

  /** \brief Gets the fully observant agents and their conditions. */
  [[nodiscard]] const ObservabilitiesMap &get_fully_observants() const;

  /** \brief Gets the partially observant agents and their conditions. */
  [[nodiscard]] const ObservabilitiesMap &get_partially_observants() const;

  /** \brief Gets the non-owning pointer to the corresponding grounded plank
   * action (only populated by PlankTranslator; null in legacy/test code
   * paths). Used by the non-mA* transition function when full-DEL semantics
   * are required. The pointee is owned by Domain's PlankPipeline. */
  [[nodiscard]] const plank::del::action *get_del_action() const noexcept {
    return m_del_action;
  }

  /** \brief Sets the non-owning del::action pointer. Called by
   * PlankTranslator::populate_actions. */
  void set_del_action(const plank::del::action *p) noexcept {
    m_del_action = p;
  }
  ///@}

  /// \name Population from translated input
  ///@{
  /** \brief Adds an executability condition (already converted). */
  void add_executability(const BeliefFormula &to_add);

  /** \brief Adds an effect with its condition (already grounded/converted). */
  void add_effect(const FluentFormula &to_add, const BeliefFormula &condition);

  /** \brief Marks an agent as fully observant under a condition. */
  void add_fully_observant(const Agent &ag, const BeliefFormula &condition);

  /** \brief Marks an agent as partially observant under a condition. */
  void add_partially_observant(const Agent &ag, const BeliefFormula &condition);
  ///@}

  /// \name Main Methods
  ///@{
  /** \brief Prints this action.*/
  void print() const;

  /** \brief Operator < implemented to use Action in std::set. */
  bool operator<(const Action &) const;

  /** \brief Human-readable string for a PropositionType. */
  static std::string type_to_string(PropositionType type);
  ///@}

private:
  /// \name Fields
  ///@{
  std::string m_name; ///< The name of this action.
  ActionId m_id; ///< The unique id of this action (calculated with grounder).
  Agent m_executor; ///< The agent that executes the action.
  PropositionType m_type =
      PropositionType::NOTSET; ///< The proposition type of this action.

  FormulaeList m_executability; ///< Executability conditions.
  ObservabilitiesMap
      m_fully_observants; ///< Fully observant agents and their conditions.
  ObservabilitiesMap m_partially_observants; ///< Partially observant agents and
                                             ///< their conditions.
  EffectsMap m_effects;                      ///< Effects and their conditions.

  /** Non-owning. Populated by PlankTranslator::populate_actions; alive for as
   * long as Domain (which owns the PlankPipeline). */
  const plank::del::action *m_del_action = nullptr;
  ///@}
};

/// \brief A set of Action objects.
using ActionsSet = std::set<Action>;

/// \brief A sequential execution of Action objects.
using ActionList = std::vector<Action>;
