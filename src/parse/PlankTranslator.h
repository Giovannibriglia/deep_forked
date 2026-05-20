/**
 * \class PlankTranslator
 * \brief Translates the grounded `del::planning_task` produced by plank
 *        into the runtime objects consumed by deep's Domain.
 *
 * The translator is stateless apart from holding a reference to the
 * `PlankPipeline` that owns the AST and the grounder info. Each public
 * method writes into a Domain field passed by reference, mirroring the
 * shape of the old `Domain::build_*` calls so that Domain.cpp stays the
 * sole orchestrator.
 *
 * Mapping policy (locked in design):
 *   - Action-type names (`public-ontic`, `public-sensing`, `semi-private-sensing`,
 *     `public-announcement`, `semi-private-announcement`) drive
 *     `PropositionType`; any other action-type is a hard error.
 *   - Observability types: only `Fully`/`Partially` are accepted; anything else
 *     is a hard error.
 *   - `(:facts-init ...)` rows become immutable fluents added to the fluent
 *     set and asserted in the initial pointed condition.
 *   - The initial state is translated from the AST `(:init ...)` formulas
 *     (each finitary-S5 entry), not from the grounded Kripke state.
 */

#pragma once

#include "PlankPipeline.h"
#include "actions/Action.h"
#include "domain/Grounder.h"
#include "domain/InitialStateInformation.h"
#include "formulae/BeliefFormula.h"
#include "utilities/Define.h"

#include "del/language/formulas.h"

#include <vector>

class PlankTranslator {
public:
  explicit PlankTranslator(PlankPipeline &pipeline) noexcept
      : m_pipeline(pipeline) {}

  /// Build the agent set and name→bitset map from `language->get_agents`.
  void build_agents(AgentsSet &agents, AgentsMap &agent_map) const;

  /// Build the fluent set, positive-fluent vector, and name→bitset map
  /// (positive + negated entries) from `language->get_atoms`. Plank's atoms
  /// include grounded predicates *and* facts (`language->is_fact(atom)`);
  /// facts are added as ordinary fluents per the mapping policy.
  void build_fluents(FluentsSet &fluents, std::vector<Fluent> &positive_fluents,
                     FluentMap &fluent_map) const;

  /// Build empty Action shells for every name in `task.actions_names` and the
  /// name→ActionId map. Call `populate_actions` afterwards to fill effects /
  /// observability — that step needs the Grounder already wired into
  /// HelperPrint.
  void build_actions(ActionsSet &actions,
                     ActionNamesMap &action_name_map) const;

  /// Walk plank's grounded `task.actions` and fill in PropositionType,
  /// effects, executability, and per-agent observability for each Action in
  /// `actions`. Replaces the old `build_propositions` loop.
  void populate_actions(ActionsSet &actions, const Grounder &grounder) const;

  /// Translate the AST `(:init ...)` formulas into deep's initial-description.
  /// Top-level propositional/atomic formulas become pointed-world conditions;
  /// modal formulas become initial belief conditions. Also asserts every
  /// `(:facts-init ...)` row in the pointed condition.
  void build_initial(InitialStateInformation &initial,
                     const Grounder &grounder) const;

  /// Translate `task.goal` (already grounded) into a CNF FormulaeList.
  void build_goal(FormulaeList &goal, const Grounder &grounder) const;

  /// Convert a plank grounded formula into deep's BeliefFormula tree.
  /// Public so that other translation steps can reuse it.
  [[nodiscard]] BeliefFormula
  convert_formula(const plank::del::formula_ptr &f,
                  const Grounder &grounder) const;

private:
  PlankPipeline &m_pipeline;

  // Lookup the deep Fluent bitset for a plank atom id (positive form).
  [[nodiscard]] Fluent atom_to_fluent(plank::del::atom a,
                                      const Grounder &grounder) const;
  // Lookup the deep Agent bitset for a plank agent id.
  [[nodiscard]] Agent agent_to_agent(plank::del::agent a,
                                     const Grounder &grounder) const;
  // Convert plank's `agent_set` (bit_deque) into deep's AgentsSet.
  [[nodiscard]] AgentsSet
  agent_set_to_agents(const plank::del::agent_set &ags,
                      const Grounder &grounder) const;

  // Map a plank action-type name to a deep PropositionType. Hard-errors on
  // unknown names.
  [[nodiscard]] static PropositionType
  action_type_to_proposition(const std::string &action_type_name,
                             const std::string &action_name);
};
