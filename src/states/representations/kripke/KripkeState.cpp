/*
 * \brief Implementation of \ref KripkeState.h
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 17, 2025
 */

#include <boost/dynamic_bitset.hpp>
#include <iostream>
#include <set>
#include <tuple>

#include "ArgumentParser.h"
#include "Domain.h"
#include "FormulaHelper.h"
#include "HelperPrint.h"
#include "InitialStateInformation.h"
#include "KripkeEntailmentHelper.h"
#include "KripkeReachabilityHelper.h"
#include "KripkeState.h"

#include <ranges>
#include <unordered_set>

#include "KripkeEqualityHelper.h"
#include "KripkeStorage.h"
#include "SetHelper.h"
#include "utilities/ExitHandler.h"

#ifdef USE_NEURALNETS
#include "neuralnets/GraphNN.h"
#endif

#ifndef USE_MASTAR
// Full-DEL transition path needs plank's grounded action + language types.
#include "del/language/formulas.h"
#include "del/language/language.h"
#include "del/semantics/actions/action.h"
#include "del/semantics/states/state.h"
#include "epddl/ast/problems/init/initial_state_decl_ast.h"
#include "epddl/ast/problems/init/explicit_initial_state_ast.h"
#include "parse/PlankPipeline.h"
#include <deque>
#include <map>
#include <type_traits>
#include <variant>
#endif

// --- Setters ---

void KripkeState::set_worlds(const KripkeWorldPointersSet &to_set) {
  m_worlds = to_set;
  if (ArgumentParser::get_instance().get_strong_equality()) {
    set_worlds_vec();
  }
}

void KripkeState::set_worlds_vec() {
  m_worlds_vec = KripkeEqualityHelper::canonicalize_worlds(m_worlds);
}

void KripkeState::set_pointed(const KripkeWorldPointer &to_set) {
  m_pointed = to_set;
  m_designated_worlds.clear();
  m_designated_worlds.insert(to_set);
}

void KripkeState::set_designated_worlds(
    const KripkeWorldPointersSet &to_set) {
  if (to_set.empty()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::ActionTypeConflict,
        "KripkeState::set_designated_worlds called with an empty set.");
  }
  m_designated_worlds = to_set;
  // Canonical pick: smallest by KripkeWorldPointer::operator< (std::set is
  // already sorted, so begin() is smallest).
  m_pointed = *m_designated_worlds.begin();
}

void KripkeState::set_beliefs(const KripkeWorldPointersTransitiveMap &to_set) {
  m_beliefs = to_set;
  if (ArgumentParser::get_instance().get_strong_equality()) {
    set_beliefs_vec();
  }
}
void KripkeState::set_beliefs_vec() {
  m_beliefs_vec = KripkeEqualityHelper::canonicalize_transitive_map(m_beliefs);
}

void KripkeState::clear_beliefs() { m_beliefs.clear(); }

void KripkeState::set_max_depth(const unsigned int to_set) noexcept {
  if (m_max_depth < to_set)
    m_max_depth = to_set;
}

// --- Getters ---

[[nodiscard]] const KripkeWorldPointersSet &
KripkeState::get_worlds() const noexcept {
  return m_worlds;
}

const KripkeWorldPointersVec &KripkeState::get_worlds_vec() const noexcept {
  return m_worlds_vec;
}

[[nodiscard]] const KripkeWorldPointer &
KripkeState::get_pointed() const noexcept {
  return m_pointed;
}

[[nodiscard]] const KripkeWorldPointersSet &
KripkeState::get_designated_worlds() const noexcept {
  return m_designated_worlds;
}

[[nodiscard]] bool KripkeState::is_multi_pointed() const noexcept {
  return m_designated_worlds.size() > 1;
}

[[nodiscard]] const KripkeWorldPointersTransitiveMap &
KripkeState::get_beliefs() const noexcept {
  return m_beliefs;
}

const KripkeWorldPointersTransitiveMapVec &
KripkeState::get_beliefs_vec() const noexcept {
  return m_beliefs_vec;
}

[[nodiscard]] unsigned int KripkeState::get_max_depth() const noexcept {
  return m_max_depth;
}

// --- Operators ---

KripkeState &KripkeState::operator=(const KripkeState &to_copy) {
  set_worlds(to_copy.get_worlds());
  set_beliefs(to_copy.get_beliefs());
  m_max_depth = to_copy.get_max_depth();
  // Copy designated set directly so we preserve multi-pointed structure;
  // set_pointed/set_designated_worlds would clobber it through the
  // single-pointed setter path.
  m_designated_worlds = to_copy.get_designated_worlds();
  m_pointed = to_copy.get_pointed();
  return *this;
}

bool KripkeState::operator==(const KripkeState &to_compare) const {
  return !((*this) < to_compare) && !(to_compare < (*this));
}

bool KripkeState::operator<(const KripkeState &to_compare) const {
  if (ArgumentParser::get_instance().get_strong_equality()) {
    return KripkeEqualityHelper::strong_less_operator(*this, to_compare);
  }
  return KripkeEqualityHelper::shallow_less_operator(*this, to_compare);
}

void KripkeState::print() const {
  HelperPrint::get_instance().print_state(*this);
}

void KripkeState::print_dot_format(std::ofstream &ofs) const {
  HelperPrint::get_instance().print_dot_format(*this, ofs);
}

void KripkeState::print_dataset_format(std::ofstream &ofs) const {
  HelperPrint::print_dataset_format(*this, ofs);
}

// --- Structure Building ---

void KripkeState::add_world(const KripkeWorld &to_add) {
  m_worlds.insert(KripkeStorage::get_instance().add_world(to_add));
}

KripkeWorldPointer KripkeState::add_rep_world(const KripkeWorld &to_add,
                                              const unsigned short repetition,
                                              bool &is_new) {
  KripkeWorldPointer tmp = KripkeStorage::get_instance().add_world(to_add);
  tmp.set_repetition(repetition);
  is_new = std::get<1>(m_worlds.insert(tmp));
  return tmp;
}

KripkeWorldPointer
KripkeState::add_rep_world(const KripkeWorld &to_add,
                           const unsigned short old_repetition) {
  bool tmp = false;
  return add_rep_world(to_add, get_max_depth() + old_repetition, tmp);
}

KripkeWorldPointer KripkeState::add_rep_world(const KripkeWorld &to_add) {
  bool tmp = false;
  return add_rep_world(to_add, get_max_depth(), tmp);
}

void KripkeState::add_edge(const KripkeWorldPointer &from,
                           const KripkeWorldPointer &to, const Agent &ag) {
  auto from_beliefs = m_beliefs.find(from);
  if (from_beliefs != m_beliefs.end()) {
    auto &beliefs_map = from_beliefs->second;
    auto ag_beliefs = beliefs_map.find(ag);
    if (ag_beliefs != beliefs_map.end()) {
      ag_beliefs->second.insert(to);
    } else {
      beliefs_map.emplace(ag, KripkeWorldPointersSet{to});
    }
  } else {
    KripkeWorldPointersMap pwm;
    pwm.emplace(ag, KripkeWorldPointersSet{to});
    m_beliefs.emplace(from, std::move(pwm));
  }
}

void KripkeState::add_world_beliefs(const KripkeWorldPointer &world,
                                    const KripkeWorldPointersMap &beliefs) {
  m_beliefs[world] = beliefs;
  /**TEMPORARY PATCH**/
  for (const auto &to_add : beliefs | std::views::values) {
    for (const auto &pw : to_add) {
      m_worlds.insert(pw);
      // bool is_new = false;
      //  add_rep_world(KripkeWorld(pw.get_fluent_set()), pw.get_repetition(),
      //  is_new);
    }
  }
  /**END TEMPORARY PATCH**/
}

void KripkeState::build_initial() {
#ifndef USE_MASTAR
  // Under DEL, plank already grounded the (:init …) into task().initial_state
  // regardless of whether it was a finitary-S5 theory or an explicit Kripke
  // structure. Reuse that grounded state instead of redoing the enumeration
  // on deep's side — deep's generate_initial_worlds is O(2^|unknown_fluents|)
  // and blows up on any non-trivial domain (Selective-Communication has 46
  // fluents → 2^37 permutations).
  if (const auto *pipeline = Domain::get_instance().get_pipeline()) {
    if (pipeline->task().initial_state != nullptr) {
      build_initial_from_plank_state();
      return;
    }
  }
#endif

  FluentsSet permutation;
  const InitialStateInformation ini_conditions =
      Domain::get_instance().get_initial_description();
  generate_initial_worlds(permutation, 0,
                          ini_conditions.get_initially_known_fluents());
  generate_initial_edges();
  // add_initial_world assigns m_pointed directly (bypassing set_pointed) so
  // the designated-worlds invariant has not been established yet. Restore it
  // now: in the initial state we are single-pointed (the unique world matching
  // the pointed-world conditions). Without this, multi-pointed entailment
  // (KripkeEntailmentHelper::entails over an empty designated set) collapses
  // to vacuously-true and every goal looks satisfied at the initial state.
  m_designated_worlds.clear();
  m_designated_worlds.insert(m_pointed);
}

void KripkeState::generate_initial_worlds(FluentsSet &permutation,
                                          const unsigned int index,
                                          const FluentsSet &initially_known) {
  auto const fluent_number = Domain::get_instance().get_fluent_number();
  auto const bit_size = Domain::get_instance().get_size_fluent();

  if (index == fluent_number) {
    const KripkeWorld to_add(permutation);
    add_initial_world(to_add);
    return;
  }

  FluentsSet permutation_2 = permutation;
  boost::dynamic_bitset<> bitSetToFindPositive(bit_size, index);
  boost::dynamic_bitset<> bitSetToFindNegative(bit_size, index);
  bitSetToFindNegative.set(bitSetToFindPositive.size() - 1, true);
  bitSetToFindPositive.set(bitSetToFindPositive.size() - 1, false);

  if (!initially_known.contains(bitSetToFindNegative)) {
    permutation.insert(bitSetToFindPositive);
    generate_initial_worlds(permutation, index + 1, initially_known);
  }
  if (!initially_known.contains(bitSetToFindPositive)) {
    permutation_2.insert(bitSetToFindNegative);
    generate_initial_worlds(permutation_2, index + 1, initially_known);
  }
}

void KripkeState::add_initial_world(const KripkeWorld &possible_add) {
  const InitialStateInformation ini_conditions =
      Domain::get_instance().get_initial_description();
  const auto &ff_forS5 = ini_conditions.get_ff_forS5();
  FluentFormula ff_forS5_nonempty;
  for (const auto &s : ff_forS5) {
    if (!s.empty()) {
      ff_forS5_nonempty.insert(s);
    }
  }
  if (ff_forS5_nonempty.empty() ||
      KripkeEntailmentHelper::entails(ff_forS5_nonempty, possible_add)) {
    add_world(possible_add);
    if (KripkeEntailmentHelper::entails(
            ini_conditions.get_pointed_world_conditions(), possible_add)) {
      m_pointed = KripkeWorldPointer(possible_add);
    }
  } else {
    KripkeStorage::get_instance().add_world(possible_add);
  }
}

void KripkeState::generate_initial_edges() {
  for (auto it_pwps_1 = m_worlds.begin(); it_pwps_1 != m_worlds.end();
       ++it_pwps_1) {
    for (auto it_pwps_2 = it_pwps_1; it_pwps_2 != m_worlds.end(); ++it_pwps_2) {
      for (const auto &agent : Domain::get_instance().get_agents()) {
        add_edge(*it_pwps_1, *it_pwps_2, agent);
        add_edge(*it_pwps_2, *it_pwps_1, agent);
      }
    }
  }

  const auto &ini_conditions = Domain::get_instance().get_initial_description();
  for (const auto &bf : ini_conditions.get_initial_conditions()) {
    remove_initial_edge_bf(bf);
  }
}

void KripkeState::remove_edge(const KripkeWorldPointer &from,
                              const KripkeWorldPointer &to, const Agent &ag) {
  auto from_beliefs = m_beliefs.find(from);
  if (from_beliefs != m_beliefs.end()) {
    auto ag_beliefs = from_beliefs->second.find(ag);
    if (ag_beliefs != from_beliefs->second.end()) {
      ag_beliefs->second.erase(to);
    }
  }
}

void KripkeState::remove_initial_edge(const FluentFormula &known_ff,
                                      const Agent &ag) {
  for (const auto &pwptr_tmp1 : m_worlds) {
    for (const auto &pwptr_tmp2 : m_worlds) {
      if (pwptr_tmp1 == pwptr_tmp2)
        continue;
      const bool entails1 =
          KripkeEntailmentHelper::entails(known_ff, pwptr_tmp1);
      const bool entails2 =
          KripkeEntailmentHelper::entails(known_ff, pwptr_tmp2);
      if (entails1 && !entails2) {
        remove_edge(pwptr_tmp1, pwptr_tmp2, ag);
        remove_edge(pwptr_tmp2, pwptr_tmp1, ag);
      } else if (entails2 && !entails1) {
        remove_edge(pwptr_tmp2, pwptr_tmp1, ag);
        remove_edge(pwptr_tmp1, pwptr_tmp2, ag);
      }
    }
  }
}

void KripkeState::remove_initial_edge_bf(const BeliefFormula &to_check) {
  if (to_check.get_formula_type() == BeliefFormulaType::C_FORMULA) {
    const BeliefFormula &tmp = to_check.get_bf1();
    switch (tmp.get_formula_type()) {
    case BeliefFormulaType::PROPOSITIONAL_FORMULA:
      if (tmp.get_operator() == BeliefFormulaOperator::BF_OR) {
        auto known_ff_ptr = FluentFormula();
        FormulaHelper::check_Bff_notBff(tmp.get_bf1(), tmp.get_bf2(),
                                        known_ff_ptr);
        if (!known_ff_ptr.empty()) {
          remove_initial_edge(known_ff_ptr, tmp.get_bf2().get_agent());
        }
      } else if (tmp.get_operator() != BeliefFormulaOperator::BF_AND) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::FormulaBadDeclaration,
            "Error: Invalid type of initial formula (FIFTH) in "
            "remove_initial_edge_bf.");
      }
      break;
    case BeliefFormulaType::FLUENT_FORMULA:
    case BeliefFormulaType::BELIEF_FORMULA:
    case BeliefFormulaType::BF_EMPTY:
      return;
    default:
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::FormulaBadDeclaration,
          "Error: Invalid type of initial formula (SIXTH) in "
          "remove_initial_edge_bf.");
    }
  } else {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::FormulaBadDeclaration,
                                   "Error: Invalid type of initial formula "
                                   "(SEVENTH) in remove_initial_edge_bf.");
  }
}

void KripkeState::compact_repetitions() {
  // 1) Collect unique labels
  if (get_max_depth() < LIMIT_REP)
    return;

  KripkeState old;
#ifdef DEBUG
  // if (ArgumentParser::get_instance().get_verbose())
  { old = *this; }
#endif

  std::vector<unsigned short> uniq;
  uniq.reserve(m_worlds.size());
  {
    std::unordered_set<unsigned short> seen;
    for (const auto &w : m_worlds) {
      auto curr_repetition = w.get_repetition();
      if (seen.insert(curr_repetition).second)
        uniq.push_back(curr_repetition);
    }
  }

  // 2) Sort and build mapping old -> rank [0..(#unique-1)]
  std::ranges::sort(uniq);
  std::unordered_map<unsigned short, unsigned short> remap;
  remap.reserve(uniq.size());
  for (unsigned short i = 0; i < static_cast<unsigned short>(uniq.size());
       ++i) {
    remap.emplace(uniq[i], i);
  }

  // 3) Rewrite labels (copy-and-replace)

  // Keep the old pointed world to remap it to its new instance afterward
  auto pointed_old = m_pointed;

  // Worlds (set) — rebuild a new set and also track old->new object mapping
  KripkeWorldPointersSet updated_w;

  for (const auto &w : m_worlds) {
    auto w2 = w; // copy the element
    // remap repetition (present by construction because uniq came from
    // m_worlds)
    if (auto itRem = remap.find(w.get_repetition()); itRem != remap.end()) {
      w2.set_repetition(itRem->second);
    } else {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::GNNBitmaskRepetitionError,
          "Error: In Compacting the repetition found mismatch (1)");
    }
    updated_w.insert(w2);
  }

  // Replace worlds
  m_worlds = std::move(updated_w);

  // Fallback: just remap its repetition if it wasn't among m_worlds
  if (auto itRem = remap.find(m_pointed.get_repetition());
      itRem != remap.end()) {
    m_pointed.set_repetition(itRem->second);
  } else {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNBitmaskRepetitionError,
        "Error: In Compacting the repetition found mismatch (2)");
  }

  // Designated worlds — same remap as m_worlds. Without this, multi-pointed
  // states leak stale repetition indices into m_designated_worlds and the
  // entries no longer match m_worlds.
  KripkeWorldPointersSet updated_d;
  for (const auto &dw : m_designated_worlds) {
    auto dw2 = dw;
    if (auto itRem = remap.find(dw.get_repetition()); itRem != remap.end()) {
      dw2.set_repetition(itRem->second);
    } else {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::GNNBitmaskRepetitionError,
          "Error: In Compacting the repetition found mismatch (designated)");
    }
    updated_d.insert(dw2);
  }
  m_designated_worlds = std::move(updated_d);

  // Edges — Transitive map (copy-and-replace)
  KripkeWorldPointersTransitiveMap updated_b;

  for (const auto &[from, b_snd] : m_beliefs) {
    auto from2 = from;
    if (auto itRem = remap.find(from.get_repetition()); itRem != remap.end()) {
      from2.set_repetition(itRem->second);
    } else {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::GNNBitmaskRepetitionError,
          "Error: In Compacting the repetition found mismatch (3)");
    }

    KripkeWorldPointersMap updated_m;

    for (const auto &[ag, m_snd] : b_snd) {
      auto &dest_set = updated_m[ag];
      for (const auto &to : m_snd) {
        auto to2 = to; // copy the element
        // remap repetition (present by construction because uniq came from
        // m_worlds)
        if (auto itRem = remap.find(to.get_repetition());
            itRem != remap.end()) {
          to2.set_repetition(itRem->second);
        } else {
          ExitHandler::exit_with_message(
              ExitHandler::ExitCode::GNNBitmaskRepetitionError,
              "Error: In Compacting the repetition found mismatch (4)");
        }
        dest_set.insert(to2);
      }
    }

    updated_b[from2] = std::move(updated_m);
  }

  // Replace transitive map
  m_beliefs = std::move(updated_b);

  // Max Depth
  m_max_depth = static_cast<unsigned int>(uniq.size());

#ifdef DEBUG
  if (ArgumentParser::get_instance().get_verbose()) {
    auto &os = ArgumentParser::get_instance().get_output_stream();

    os << "[COMPACT_REP]";
    FormulaHelper::checkSameKState(*this, old);
  }
#endif
}

// --- Transition/Execution ---

#ifndef USE_MASTAR
namespace {
namespace pd = plank::del;

/// Lookup tables built once per transition: plank atom/agent ids → deep
/// Fluent/Agent bitsets. Same indexing as plank's language object.
struct DelCaches {
  std::vector<Fluent> pos_fluent; ///< indexed by plank atom id
  std::vector<Fluent> neg_fluent; ///< indexed by plank atom id
  std::vector<Agent> agent;       ///< indexed by plank agent id
};

DelCaches build_caches(const pd::language &lang, const Grounder &g) {
  DelCaches c;
  const auto an = lang.get_atoms_number();
  c.pos_fluent.reserve(an);
  c.neg_fluent.reserve(an);
  for (pd::atom a = 0; a < an; ++a) {
    const auto &name = lang.get_atom_name(a);
    c.pos_fluent.push_back(g.ground_fluent(name));
    c.neg_fluent.push_back(g.ground_fluent(NEGATION_SYMBOL + name));
  }
  const auto agn = lang.get_agents_number();
  c.agent.reserve(agn);
  for (pd::agent ag = 0; ag < agn; ++ag) {
    c.agent.push_back(g.ground_agent(lang.get_agent_name(ag)));
  }
  return c;
}

/// Recursively evaluate a plank grounded formula at a world inside a state.
/// Modal operators traverse `S.get_beliefs()`; common knowledge does a BFS
/// over the union of agent edges in the group.
bool eval(const pd::formula_ptr &f, const KripkeWorldPointer &w,
          const KripkeState &S, const DelCaches &c) {
  return std::visit(
      [&](auto &&ptr) -> bool {
        using T = std::decay_t<decltype(ptr)>;
        if constexpr (std::is_same_v<T, pd::true_formula_ptr>) {
          return true;
        } else if constexpr (std::is_same_v<T, pd::false_formula_ptr>) {
          return false;
        } else if constexpr (std::is_same_v<T, pd::atom_formula_ptr>) {
          const auto &fs = w.get_fluent_set();
          return fs.find(c.pos_fluent[ptr->get_atom()]) != fs.end();
        } else if constexpr (std::is_same_v<T, pd::not_formula_ptr>) {
          return !eval(ptr->get_formula(), w, S, c);
        } else if constexpr (std::is_same_v<T, pd::and_formula_ptr>) {
          for (const auto &sub : ptr->get_formulas())
            if (!eval(sub, w, S, c)) return false;
          return true;
        } else if constexpr (std::is_same_v<T, pd::or_formula_ptr>) {
          for (const auto &sub : ptr->get_formulas())
            if (eval(sub, w, S, c)) return true;
          return false;
        } else if constexpr (std::is_same_v<T, pd::imply_formula_ptr>) {
          return !eval(ptr->get_first_formula(), w, S, c) ||
                 eval(ptr->get_second_formula(), w, S, c);
        } else if constexpr (std::is_same_v<T, pd::box_formula_ptr> ||
                             std::is_same_v<T, pd::diamond_formula_ptr>) {
          const bool universal = std::is_same_v<T, pd::box_formula_ptr>;
          const auto bit = S.get_beliefs().find(w);
          if (bit == S.get_beliefs().end()) return universal;
          for (auto ai : ptr->get_mod_index()) {
            auto eit = bit->second.find(c.agent[ai]);
            if (eit == bit->second.end()) continue;
            for (const auto &wp : eit->second) {
              const bool sub = eval(ptr->get_formula(), wp, S, c);
              if (universal && !sub) return false;
              if (!universal && sub) return true;
            }
          }
          return universal;
        } else if constexpr (std::is_same_v<T, pd::kw_box_formula_ptr> ||
                             std::is_same_v<T, pd::kw_diamond_formula_ptr>) {
          // Kw_G phi == [G] phi OR [G] !phi;  <Kw_G> phi == ![G]phi AND ![G]!phi
          auto box_with_target = [&](bool tgt) {
            const auto bit = S.get_beliefs().find(w);
            if (bit == S.get_beliefs().end()) return true;
            for (auto ai : ptr->get_mod_index()) {
              auto eit = bit->second.find(c.agent[ai]);
              if (eit == bit->second.end()) continue;
              for (const auto &wp : eit->second)
                if (eval(ptr->get_formula(), wp, S, c) != tgt) return false;
            }
            return true;
          };
          if constexpr (std::is_same_v<T, pd::kw_box_formula_ptr>)
            return box_with_target(true) || box_with_target(false);
          else
            return !box_with_target(true) && !box_with_target(false);
        } else if constexpr (std::is_same_v<T, pd::c_box_formula_ptr> ||
                             std::is_same_v<T, pd::c_diamond_formula_ptr>) {
          // BFS over the union of agent edges in the group; closed under
          // reflexive-transitive S5 reachability (start from w).
          AgentsSet ags;
          for (auto ai : ptr->get_mod_index()) ags.insert(c.agent[ai]);
          KripkeWorldPointersSet visited;
          std::deque<KripkeWorldPointer> q;
          visited.insert(w);
          q.push_back(w);
          while (!q.empty()) {
            auto cur = q.front();
            q.pop_front();
            auto bit = S.get_beliefs().find(cur);
            if (bit == S.get_beliefs().end()) continue;
            for (const auto &ag : ags) {
              auto eit = bit->second.find(ag);
              if (eit == bit->second.end()) continue;
              for (const auto &wp : eit->second) {
                if (visited.insert(wp).second) q.push_back(wp);
              }
            }
          }
          if constexpr (std::is_same_v<T, pd::c_box_formula_ptr>) {
            for (const auto &wp : visited)
              if (!eval(ptr->get_formula(), wp, S, c)) return false;
            return true;
          } else {
            for (const auto &wp : visited)
              if (eval(ptr->get_formula(), wp, S, c)) return true;
            return false;
          }
        }
        return false;
      },
      f);
}

} // anonymous namespace

KripkeState KripkeState::compute_successor_del(const Action &act) const {
  const KripkeState &S = *this;
  const pd::action &pa = *act.get_del_action();
  // (Null check is performed by the caller — compute_successor.)

  const auto *pipeline = Domain::get_instance().get_pipeline();
  if (pipeline == nullptr) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::ActionTypeConflict,
        "DEL transition requires Domain's PlankPipeline; build() must run "
        "before any compute_successor call.");
  }
  const pd::language &lang = pipeline->language();
  const Grounder &g = HelperPrint::get_instance().get_grounder();
  const DelCaches caches = build_caches(lang, g);

  const auto n_events = pa.get_events_number();
  const auto n_agents = lang.get_agents_number();
  const auto n_obs_types = pa.get_obs_types_number();

  // 1. Build new worlds W' = { (w, e) | pre(e) holds at w }. Postconditions
  //    are evaluated at the *input* world (DEL semantics).
  using WEKey = std::pair<KripkeWorldId, pd::event_id>;
  std::map<WEKey, KripkeWorldPointer> new_map;

  KripkeState ret;
  ret.set_max_depth(S.get_max_depth() + 1);

  // Repetition tags act as the world identity discriminator: a
  // KripkeWorldPointer hashes (underlying fluent-id, repetition) into its
  // own id, so two pointers with the same fluent valuation are equal iff
  // they share a repetition. The DEL product (w, e) needs each (input world,
  // event) pair to be distinct even when events have trivial postconditions
  // (e.g. private-announcement's nil event leaves fluents untouched, so
  // (w, e_tell) and (w, nil) end up with identical fluent sets). We encode
  // the event index into the repetition so that any two (w, e) and (w, e')
  // get different pointer identities — otherwise both collapse to the same
  // m_worlds entry and the new accessibility relation cannot distinguish
  // the "fully observant" stratum from the "oblivious" stratum, which
  // silently drops every knowledge transfer.
  for (const auto &w : S.get_worlds()) {
    for (pd::event_id e = 0; e < n_events; ++e) {
      if (!eval(pa.get_precondition(e), w, S, caches)) continue;
      FluentsSet new_fluents = w.get_fluent_set();
      const auto &post = pa.get_postconditions(e);
      for (const auto &[atom_id, value_formula] : post) {
        const bool new_val = eval(value_formula, w, S, caches);
        const Fluent &p_pos = caches.pos_fluent[atom_id];
        const Fluent &p_neg = caches.neg_fluent[atom_id];
        new_fluents.erase(p_pos);
        new_fluents.erase(p_neg);
        new_fluents.insert(new_val ? p_pos : p_neg);
      }
      const auto event_rep =
          static_cast<unsigned short>(w.get_repetition() * n_events + e);
      KripkeWorldPointer new_p =
          ret.add_rep_world(KripkeWorld(new_fluents), event_rep);
      new_map.emplace(WEKey{w.get_id(), e}, new_p);
    }
  }

  // 2. Per-agent obs_type: evaluate obs_condition(i, t) at the original
  //    pointed; pick the first applicable t. plank stores obs-conditions
  //    sparsely (unordered_map<obs_type, formula>) — use the whole map and
  //    `find` rather than `get_obs_condition(ai, t)` which calls `.at(t)`
  //    and throws when the (agent, obs_type) pair is unmaterialized.
  std::vector<long> agent_obs_type(n_agents, -1);
  for (pd::agent ai = 0; ai < n_agents; ++ai) {
    const auto &obs_map = pa.get_agent_obs_conditions(ai);
    for (pd::obs_type t = 0; t < n_obs_types; ++t) {
      const auto it = obs_map.find(t);
      if (it == obs_map.end()) continue;
      const auto &cond = it->second;
      if (pd::formulas_utils::get_type(cond) == pd::formula_type::false_formula)
        continue;
      if (eval(cond, S.get_pointed(), S, caches)) {
        agent_obs_type[ai] = static_cast<long>(t);
        break;
      }
    }
  }

  // 3. New accessibility: ((w,e), (w',e')) ∈ R'_i iff
  //    (w,w') ∈ R_i in S AND (e,e') ∈ R^A_t (where t = obs_type of i).
  for (const auto &w : S.get_worlds()) {
    auto bit = S.get_beliefs().find(w);
    if (bit == S.get_beliefs().end()) continue;
    for (pd::event_id e = 0; e < n_events; ++e) {
      auto it_we = new_map.find({w.get_id(), e});
      if (it_we == new_map.end()) continue;
      for (pd::agent ai = 0; ai < n_agents; ++ai) {
        if (agent_obs_type[ai] < 0) continue;
        const auto t = static_cast<pd::obs_type>(agent_obs_type[ai]);
        const Agent &dag = caches.agent[ai];
        auto eit = bit->second.find(dag);
        if (eit == bit->second.end()) continue;
        for (const auto &wp : eit->second) {
          for (pd::event_id ep = 0; ep < n_events; ++ep) {
            if (!pa.has_edge(t, e, ep)) continue;
            auto it_we2 = new_map.find({wp.get_id(), ep});
            if (it_we2 == new_map.end()) continue;
            ret.add_edge(it_we->second, it_we2->second, dag);
          }
        }
      }
    }
  }

  // 4. New designated worlds: for every designated event e in the action
  //    whose precondition holds at one of the input designated worlds w,
  //    add (w, e) to the result's designated set. For single-pointed input
  //    + single applicable designated event (the mA* case), this collapses
  //    to a single canonical pointed; otherwise the result is multi-pointed.
  KripkeWorldPointersSet new_designated;
  for (const auto &w_d : S.get_designated_worlds()) {
    for (auto e : pa.get_designated_events()) {
      if (!eval(pa.get_precondition(e), w_d, S, caches)) continue;
      auto it = new_map.find({w_d.get_id(), e});
      if (it != new_map.end()) new_designated.insert(it->second);
    }
  }
  if (new_designated.empty()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::ActionTypeConflict,
        "DEL transition: no designated event of action '" + act.get_name() +
            "' is applicable at any input designated world.");
  }
  ret.set_designated_worlds(new_designated);
  return ret;
}

bool KripkeState::is_executable_del(const Action &act) const {
  const pd::action *pa = act.get_del_action();
  if (pa == nullptr) {
    return false;
  }
  const auto *pipeline = Domain::get_instance().get_pipeline();
  if (pipeline == nullptr) {
    return false;
  }
  const pd::language &lang = pipeline->language();
  const Grounder &g = HelperPrint::get_instance().get_grounder();
  const DelCaches caches = build_caches(lang, g);

  // Applicability under DEL: action is executable in S iff some designated
  // event has its precondition satisfied at some designated world. This is
  // exactly the condition that produces a non-empty new designated set in
  // compute_successor_del, so the two stay consistent.
  for (const auto &w_d : m_designated_worlds) {
    for (auto e : pa->get_designated_events()) {
      if (eval(pa->get_precondition(e), w_d, *this, caches)) {
        return true;
      }
    }
  }
  return false;
}

void KripkeState::build_initial_from_plank_state() {
  const auto *pipeline = Domain::get_instance().get_pipeline();
  if (pipeline == nullptr || pipeline->task().initial_state == nullptr) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::DomainInitialStateTypeError,
        "build_initial_from_plank_state called without a grounded plank "
        "initial state on the pipeline.");
  }
  const pd::state &s = *pipeline->task().initial_state;
  const pd::language &lang = pipeline->language();
  const Grounder &g = HelperPrint::get_instance().get_grounder();

  // Build atom-id → positive Fluent and agent-id → deep Agent caches. These
  // are the same caches build_caches() uses for the transition function; we
  // inline the relevant subset here to avoid pulling in the DelCaches struct.
  const auto an = lang.get_atoms_number();
  std::vector<Fluent> pos_fluent;
  std::vector<Fluent> neg_fluent;
  pos_fluent.reserve(an);
  neg_fluent.reserve(an);
  for (pd::atom a = 0; a < an; ++a) {
    const auto &name = lang.get_atom_name(a);
    pos_fluent.push_back(g.ground_fluent(name));
    neg_fluent.push_back(g.ground_fluent(NEGATION_SYMBOL + name));
  }
  const auto agn = lang.get_agents_number();
  std::vector<Agent> agent;
  agent.reserve(agn);
  for (pd::agent ag = 0; ag < agn; ++ag) {
    agent.push_back(g.ground_agent(lang.get_agent_name(ag)));
  }

  // Materialise one deep world per plank world, keyed by world_id so we can
  // translate the accessibility relation in the second pass.
  const auto worlds_n = s.get_worlds_number();
  std::vector<KripkeWorldPointer> world_index;
  world_index.reserve(static_cast<size_t>(worlds_n));
  for (pd::world_id w = 0; w < worlds_n; ++w) {
    FluentsSet fs;
    const pd::label &lab = s.get_label(w);
    for (pd::atom a = 0; a < an; ++a) {
      fs.insert(lab[a] ? pos_fluent[a] : neg_fluent[a]);
    }
    // Initial worlds are at repetition 0 by convention (same as the S5 path).
    KripkeWorldPointer wp = add_rep_world(KripkeWorld(fs), 0);
    world_index.push_back(wp);
  }

  // Translate the accessibility relation: for every (agent, source) pair, the
  // plank state exposes the bit-deque of reachable target world_ids.
  for (pd::agent ai = 0; ai < agn; ++ai) {
    for (pd::world_id w = 0; w < worlds_n; ++w) {
      const auto &reachable = s.get_agent_possible_worlds(ai, w);
      for (const auto target_w : reachable) {
        add_edge(world_index[static_cast<size_t>(w)],
                 world_index[static_cast<size_t>(target_w)], agent[ai]);
      }
    }
  }

  // Designated set comes verbatim from plank. set_designated_worlds picks the
  // canonical pointed (min by KripkeWorldPointer::operator<) and enforces the
  // non-empty invariant.
  KripkeWorldPointersSet designated;
  const auto &plank_designated = s.get_designated_worlds();
  for (const auto target_w : plank_designated) {
    designated.insert(world_index[static_cast<size_t>(target_w)]);
  }
  set_designated_worlds(designated);
}
#endif // !USE_MASTAR

KripkeState KripkeState::compute_successor(const Action &act) const {
#ifdef USE_MASTAR
  // Legacy mA* path: dispatch on the flat PropositionType built by
  // PlankTranslator. Conditional postconditions / multi-event actions are
  // rejected upstream so each Action falls into one of three categories.
  KripkeState ret;
  switch (act.get_type()) {
  case PropositionType::ONTIC:
    ret = execute_ontic(act);
    break;
  case PropositionType::SENSING:
    ret = execute_sensing(act);
    break;
  case PropositionType::ANNOUNCEMENT:
    ret = execute_announcement(act);
    break;
  default:
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::ActionTypeConflict,
        "Error: Executing an action with undefined type: " + act.get_name());
  }

  ret.compact_repetitions();
  return ret;
#else
  // Full-DEL path: consume the grounded event-based action carried by `act`
  // and run the standard DEL product update directly on this KripkeState.
  const plank::del::action *pa = act.get_del_action();
  if (pa == nullptr) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::ActionTypeConflict,
        "Action '" + act.get_name() +
            "' has no associated plank del::action; the DEL transition "
            "function requires the EPDDL pipeline.");
  }
  KripkeState ret = compute_successor_del(act);
  ret.compact_repetitions();
  return ret;
#endif
}

void KripkeState::maintain_oblivious_believed_worlds(
    KripkeState &ret, const AgentsSet &oblivious_obs_agents) const {
  if (!oblivious_obs_agents.empty()) {
    const auto tmp_world_set = KripkeReachabilityHelper::get_E_reachable_worlds(
        oblivious_obs_agents, get_pointed(), *this);
    KripkeWorldPointersSet world_oblivious;
    /*for (const auto &agent : Domain::get_instance().get_agents()) {
      for (const auto &wo_ob : tmp_world_set) {
        SetHelper::sum_set<KripkeWorldPointer>(
            world_oblivious, KripkeReachabilityHelper::get_B_reachable_worlds(
                                 agent, wo_ob, *this));
      }
    }*/
    KripkeReachabilityHelper::get_E_reachable_worlds_recursive(
        Domain::get_instance().get_agents(), tmp_world_set, world_oblivious,
        *this);

    SetHelper::sum_set<KripkeWorldPointer>(world_oblivious, tmp_world_set);
    ret.set_max_depth(get_max_depth() + 1);
    ret.set_worlds(world_oblivious);

    for (const auto &wo_ob : world_oblivious) {
      auto it_pwmap = m_beliefs.find(wo_ob);
      if (it_pwmap != m_beliefs.end()) {
        ret.add_world_beliefs(wo_ob, it_pwmap->second);
      }
    }
  }
}

KripkeWorldPointer KripkeState::execute_ontic_helper(
    const Action &act, KripkeState &ret, const KripkeWorldPointer &current_pw,
    TransitionMap &calculated, AgentsSet &oblivious_obs_agents) const {
  FluentFormula current_pw_effects =
      FormulaHelper::get_effects_if_entailed(act.get_effects(), *this);
  FluentsSet world_description = current_pw.get_fluent_set();
  for (const auto &effect : current_pw_effects) {
    FormulaHelper::apply_effect(effect, world_description);
  }

  KripkeWorldPointer new_pw = ret.add_rep_world(KripkeWorld(world_description),
                                                current_pw.get_repetition());
  calculated.insert(TransitionMap::value_type(current_pw, new_pw));

  auto it_pwtm = get_beliefs().find(current_pw);

  if (it_pwtm != get_beliefs().end()) {
    for (const auto &[ag, beliefs] : it_pwtm->second) {
      bool is_oblivious_obs = oblivious_obs_agents.contains(ag);

      for (const auto &belief : beliefs) {
        if (is_oblivious_obs) {
          auto maintained_world = ret.get_worlds().find(belief);
          if (maintained_world != ret.get_worlds().end()) {
            ret.add_edge(new_pw, belief, ag);
          }
        } else {
          auto calculated_world = calculated.find(belief);
          if (calculated_world != calculated.end()) {
            ret.add_edge(new_pw, calculated_world->second, ag);
          } else {
            KripkeWorldPointer believed_pw = execute_ontic_helper(
                act, ret, belief, calculated, oblivious_obs_agents);
            ret.add_edge(new_pw, believed_pw, ag);
            ret.set_max_depth(ret.get_max_depth() + 1 +
                              current_pw.get_repetition());
          }
        }
      }
    }
  }

  return new_pw;
}

KripkeState KripkeState::execute_ontic(const Action &act) const {
  KripkeState ret;

  AgentsSet agents = Domain::get_instance().get_agents();
  AgentsSet fully_obs_agents =
      FormulaHelper::get_agents_if_entailed(act.get_fully_observants(), *this);

  AgentsSet oblivious_obs_agents = agents;
  SetHelper::minus_set<Agent>(oblivious_obs_agents, fully_obs_agents);

  TransitionMap calculated;
  maintain_oblivious_believed_worlds(ret, oblivious_obs_agents);

  KripkeWorldPointer new_pointed = execute_ontic_helper(
      act, ret, get_pointed(), calculated, oblivious_obs_agents);
  ret.set_pointed(new_pointed);

  return ret;
}

KripkeWorldPointer KripkeState::execute_sensing_announcement_helper(
    const FluentFormula &effects, KripkeState &ret,
    const KripkeWorldPointer &current_pw, TransitionMap &calculated,
    AgentsSet &partially_obs_agents, AgentsSet &oblivious_obs_agents,
    bool previous_entailment) const {
  KripkeWorldPointer new_pw = ret.add_rep_world(
      KripkeWorld(current_pw.get_fluent_set()), current_pw.get_repetition());
  calculated.insert(TransitionMap::value_type(current_pw, new_pw));

  auto it_pwtm = get_beliefs().find(current_pw);

  if (it_pwtm != get_beliefs().end()) {
    for (const auto &[ag, beliefs] : it_pwtm->second) {
      bool is_oblivious_obs = oblivious_obs_agents.contains(ag);
      bool is_partially_obs = partially_obs_agents.contains(ag);
      bool is_fully_obs = !is_oblivious_obs && !is_partially_obs;

      for (const auto &belief : beliefs) {
        if (is_oblivious_obs) {
          auto maintained_world = ret.get_worlds().find(belief);
          if (maintained_world != ret.get_worlds().end()) {
            ret.add_edge(new_pw, belief, ag);
          }
        } else {
          auto calculated_world = calculated.find(belief);
          bool ent = KripkeEntailmentHelper::entails(effects, belief);

          bool is_consistent_belief =
              is_partially_obs ||
              (is_fully_obs && (ent == previous_entailment));

          if (calculated_world != calculated.end()) {
            if (is_consistent_belief) {
              ret.add_edge(new_pw, calculated_world->second, ag);
            }
          } else {
            if (is_consistent_belief) {
              KripkeWorldPointer believed_pw =
                  execute_sensing_announcement_helper(
                      effects, ret, belief, calculated, partially_obs_agents,
                      oblivious_obs_agents, ent);
              ret.add_edge(new_pw, believed_pw, ag);
            }
          }
        }
      }
    }
  }
  return new_pw;
}

KripkeState KripkeState::execute_sensing(const Action &act) const {
  KripkeState ret;

  AgentsSet agents = Domain::get_instance().get_agents();
  AgentsSet fully_obs_agents =
      FormulaHelper::get_agents_if_entailed(act.get_fully_observants(), *this);
  AgentsSet partially_obs_agents = FormulaHelper::get_agents_if_entailed(
      act.get_partially_observants(), *this);

  AgentsSet oblivious_obs_agents = agents;
  SetHelper::minus_set<Agent>(oblivious_obs_agents, fully_obs_agents);
  SetHelper::minus_set<Agent>(oblivious_obs_agents, partially_obs_agents);

  if (!oblivious_obs_agents.empty()) {
    ret.set_max_depth(get_max_depth() + 1);
  }

  TransitionMap calculated;
  maintain_oblivious_believed_worlds(ret, oblivious_obs_agents);

  FluentFormula effects =
      FormulaHelper::get_effects_if_entailed(act.get_effects(), *this);

  KripkeWorldPointer new_pointed = execute_sensing_announcement_helper(
      effects, ret, get_pointed(), calculated, partially_obs_agents,
      oblivious_obs_agents,
      KripkeEntailmentHelper::entails(effects, get_pointed()));
  ret.set_pointed(new_pointed);

  return ret;
}

KripkeState KripkeState::execute_announcement(const Action &act) const {
  return execute_sensing(act);
}

bool KripkeState::entails(const Fluent &to_check) const {
  return std::ranges::all_of(m_designated_worlds, [&](const auto &w) {
    return KripkeEntailmentHelper::entails(to_check, w);
  });
}

bool KripkeState::entails(const FluentsSet &to_check) const {
  return std::ranges::all_of(m_designated_worlds, [&](const auto &w) {
    return KripkeEntailmentHelper::entails(to_check, w);
  });
}

bool KripkeState::entails(const FluentFormula &to_check) const {
  return std::ranges::all_of(m_designated_worlds, [&](const auto &w) {
    return KripkeEntailmentHelper::entails(to_check, w);
  });
}

bool KripkeState::entails(const BeliefFormula &to_check) const {
  return KripkeEntailmentHelper::entails(to_check, *this);
}

bool KripkeState::entails(const FormulaeList &to_check) const {
  return KripkeEntailmentHelper::entails(to_check, *this);
}

void KripkeState::contract_with_bisimulation() {
  // Ordering invariant: `compute_successor` runs `compact_repetitions` before
  // returning, so a state reaching this point already has compacted repetition
  // labels and m_designated_worlds remapped in lockstep with m_worlds. The
  // bisimulation pass then rebuilds m_worlds / m_beliefs / m_designated_worlds
  // from scratch via automaton_to_kstate, so the two passes do not interfere.
  // Do not reorder: bisimulation must come AFTER compaction, or the contracted
  // state would carry the pre-compaction repetition labels that no longer
  // match other state slots.
  KripkeReachabilityHelper::clean_unreachable_worlds(*this);
  Bisimulation b;
  b.calc_min_bisimilar(*this);
}

const GraphTensor &KripkeState::get_tensor_representation() {
#ifdef USE_NEURALNETS
  if (!m_computed_tensor_representation) {
    m_tensor_representation =
        GraphNN<KripkeState>::get_instance().state_to_tensor_minimal(*this);
    m_computed_tensor_representation = true;
  }
  return m_tensor_representation;
#else
  ExitHandler::exit_with_message(
      ExitHandler::ExitCode::HeuristicsBadDeclaration,
      "Trying to create a tensor of a state but neural network support (onnx "
      "handler) is "
      "not "
      "enabled or linked. Please recompile with the nn option.");
  // This line will never be reached, but added to avoid compiler warning.
  std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
#endif
}

// --- Constructors ---

KripkeState::KripkeState(const KripkeState &other)
    : m_max_depth(other.m_max_depth), m_worlds(other.m_worlds),
      m_pointed(other.m_pointed),
      m_designated_worlds(other.m_designated_worlds),
      m_beliefs(other.m_beliefs), m_worlds_vec(other.m_worlds_vec),
      m_beliefs_vec(other.m_beliefs_vec) {}