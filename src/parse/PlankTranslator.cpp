#include "PlankTranslator.h"

#include "ArgumentParser.h"
#include "utilities/ExitHandler.h"
#include "utilities/FormulaHelper.h"

#include "del/language/language.h"
#include "del/semantics/actions/action.h"
#include "del/semantics/planning_task.h"
#include "epddl/ast/problems/init/finitary_s5_theory_ast.h"
#include "epddl/ast/problems/init/initial_state_decl_ast.h"
#include "epddl/ast/problems/init/facts_init_ast.h"
#include "epddl/grounder/formulas/formulas_and_lists_grounder.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <stdexcept>
#include <variant>

namespace pl = plank;

namespace {

// Wraps any grounded propositional/literal formula into a single-clause
// FluentFormula. Returns an empty FluentFormula for true_formula.
FluentFormula formula_to_fluent_formula_or_empty(const BeliefFormula &bf) {
  if (bf.get_formula_type() == BeliefFormulaType::FLUENT_FORMULA) {
    return bf.get_fluent_formula();
  }
  return {};
}

BeliefFormula true_belief_formula() {
  BeliefFormula bf;
  bf.set_formula_type(BeliefFormulaType::BF_EMPTY);
  return bf;
}

// Try to fold a BeliefFormula tree of FLUENT_FORMULA leaves and BF_AND/BF_OR
// propositional nodes into a single FluentFormula (DNF). Returns the collapsed
// FluentFormula when every leaf is FLUENT_FORMULA or BF_EMPTY and every
// internal node is BF_AND or BF_OR; std::nullopt otherwise. Modal subformulas
// (BELIEF/C/E) and BF_NOT make the tree non-collapsible — they are returned
// verbatim by convert_formula.
//
// The empty FluentFormula represents `true` (matches KripkeEntailmentHelper's
// empty-FF-is-true convention). AND with `true` is the other operand; OR with
// `true` is `true`.
std::optional<FluentFormula>
try_collapse_to_ff(const BeliefFormula &bf) {
  switch (bf.get_formula_type()) {
  case BeliefFormulaType::FLUENT_FORMULA:
    return bf.get_fluent_formula();
  case BeliefFormulaType::BF_EMPTY:
    return FluentFormula{}; // empty DNF = true
  case BeliefFormulaType::PROPOSITIONAL_FORMULA: {
    const auto op = bf.get_operator();
    if (op != BeliefFormulaOperator::BF_AND &&
        op != BeliefFormulaOperator::BF_OR &&
        op != BeliefFormulaOperator::BF_NOT) {
      return std::nullopt;
    }
    auto f1 = try_collapse_to_ff(bf.get_bf1());
    if (!f1) return std::nullopt;

    if (op == BeliefFormulaOperator::BF_NOT) {
      // DNF negation: ¬(C1 ∨ … ∨ Cn) = (¬C1) ∧ … ∧ (¬Cn), where each
      // ¬Ci = OR over negated literals. We distribute back to DNF by picking
      // one negated literal per clause across the Cartesian product.
      //   ¬(true)  → false : empty DNF means "true" in our convention; the
      //   inverse cannot be represented without ambiguity, so bail.
      if (f1->empty()) return std::nullopt;
      FluentFormula acc;
      acc.insert(FluentsSet{}); // start from {{}} = true
      for (const auto &clause : *f1) {
        if (clause.empty()) {
          // Empty clause inside a DNF would denote "false"; ¬false = true.
          return FluentFormula{};
        }
        FluentFormula next;
        for (const auto &a : acc) {
          for (const auto &lit : clause) {
            FluentsSet merged = a;
            merged.insert(FormulaHelper::negate_fluent(lit));
            next.insert(std::move(merged));
          }
        }
        acc = std::move(next);
      }
      return acc;
    }

    auto f2 = try_collapse_to_ff(bf.get_bf2());
    if (!f2) return std::nullopt;

    if (op == BeliefFormulaOperator::BF_OR) {
      // Clause union: (C1) ∨ (C2) = clauses of f1 ∪ clauses of f2.
      // Empty FF on either side denotes `true`, which absorbs the OR.
      if (f1->empty() || f2->empty()) return FluentFormula{};
      FluentFormula out = *f1;
      for (const auto &c : *f2) out.insert(c);
      return out;
    }
    // BF_AND: distribute. (C1) ∧ (C2) = { c1 ∪ c2 | c1 ∈ f1, c2 ∈ f2 }.
    // Empty FF on either side denotes `true` and is the identity for AND.
    if (f1->empty()) return f2;
    if (f2->empty()) return f1;
    FluentFormula out;
    for (const auto &c1 : *f1) {
      for (const auto &c2 : *f2) {
        FluentsSet merged = c1;
        merged.insert(c2.begin(), c2.end());
        out.insert(std::move(merged));
      }
    }
    return out;
  }
  default:
    return std::nullopt; // BELIEF / C / E / failure types: not collapsible
  }
}

} // namespace

PropositionType PlankTranslator::action_type_to_proposition(
    const std::string &action_type_name, const std::string &action_name) {
  // Fixed table over basic.epddl action-type names. Any unknown name is a
  // hard error — we will not silently map it.
  if (action_type_name == "public-ontic")
    return PropositionType::ONTIC;
  if (action_type_name == "public-sensing" ||
      action_type_name == "semi-private-sensing")
    return PropositionType::SENSING;
  if (action_type_name == "public-announcement" ||
      action_type_name == "semi-private-announcement")
    return PropositionType::ANNOUNCEMENT;

  ExitHandler::exit_with_message(
      ExitHandler::ExitCode::ParsingError,
      "Unsupported action-type '" + action_type_name + "' on action '" +
          action_name +
          "'. Supported: public-ontic, public-sensing, semi-private-sensing, "
          "public-announcement, semi-private-announcement.");
  return PropositionType::NOTSET;
}

Fluent PlankTranslator::atom_to_fluent(pl::del::atom a,
                                       const Grounder &grounder) const {
  const std::string &name = m_pipeline.language().get_atom_name(a);
  return grounder.ground_fluent(name);
}

Agent PlankTranslator::agent_to_agent(pl::del::agent a,
                                      const Grounder &grounder) const {
  const std::string &name = m_pipeline.language().get_agent_name(a);
  return grounder.ground_agent(name);
}

AgentsSet
PlankTranslator::agent_set_to_agents(const pl::del::agent_set &ags,
                                     const Grounder &grounder) const {
  AgentsSet out;
  for (auto id : ags) {
    out.insert(agent_to_agent(id, grounder));
  }
  return out;
}

void PlankTranslator::build_agents(AgentsSet &agents,
                                   AgentsMap &agent_map) const {
  auto &os = ArgumentParser::get_instance().get_output_stream();
  const bool verbose = ArgumentParser::get_instance().get_verbose();
  const auto agents_count =
      static_cast<int>(m_pipeline.language().get_agents_number());
  const int bit_size = FormulaHelper::length_to_power_two(agents_count);

  for (int i = 0; i < agents_count; ++i) {
    const std::string &name = m_pipeline.language().get_agent_name(
        static_cast<pl::del::agent>(i));
    Agent ag(bit_size, i);
    agent_map.insert({name, ag});
    agents.insert(ag);
    if (verbose) {
      os << "Agent " << name << " is " << ag << std::endl;
    }
  }
}

void PlankTranslator::build_fluents(FluentsSet &fluents,
                                    std::vector<Fluent> &positive_fluents,
                                    FluentMap &fluent_map) const {
  auto &os = ArgumentParser::get_instance().get_output_stream();
  const bool verbose = ArgumentParser::get_instance().get_verbose();
  const auto atoms_count =
      static_cast<int>(m_pipeline.language().get_atoms_number());
  const int bit_size = FormulaHelper::length_to_power_two(atoms_count) + 1;

  for (int i = 0; i < atoms_count; ++i) {
    const std::string &name = m_pipeline.language().get_atom_name(
        static_cast<pl::del::atom>(i));
    Fluent pos(bit_size, i);
    pos.set(pos.size() - 1, true);
    fluent_map.insert({name, pos});
    fluents.insert(pos);
    positive_fluents.push_back(pos);

    Fluent neg(bit_size, i);
    fluent_map.insert({NEGATION_SYMBOL + name, neg});
    fluents.insert(neg);

    if (verbose) {
      os << "Literal " << name << " is " << pos << std::endl;
      os << "Literal not " << name << " is " << neg << std::endl;
    }
  }
}

void PlankTranslator::build_actions(ActionsSet &actions,
                                    ActionNamesMap &action_name_map) const {
  auto &os = ArgumentParser::get_instance().get_output_stream();
  const bool verbose = ArgumentParser::get_instance().get_verbose();
  const auto &names = m_pipeline.task().actions_names;
  const int bit_size =
      FormulaHelper::length_to_power_two(static_cast<int>(names.size()));

  int i = 0;
  for (const auto &name : names) {
    ActionId id(bit_size, i);
    Action act(name, id);
    action_name_map.insert({name, id});
    actions.insert(act);
    if (verbose) {
      os << "Action " << name << " is " << id << std::endl;
    }
    ++i;
  }
}

void PlankTranslator::populate_actions(ActionsSet &actions,
                                       const Grounder &grounder) const {
  const auto &task = m_pipeline.task();

  for (const auto &name : task.actions_names) {
    auto it = task.actions.find(name);
    if (it == task.actions.end()) {
      ExitHandler::exit_with_message(ExitHandler::ExitCode::ParsingError,
                                     "Action '" + name +
                                         "' missing from grounded action map.");
    }
    const pl::del::action &pa = *it->second;

    // Find the matching deep Action in the set (keyed by ActionId).
    auto deep_it = std::find_if(actions.begin(), actions.end(),
                                [&name](const Action &a) {
                                  return a.get_name() == name;
                                });
    if (deep_it == actions.end()) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::ParsingError,
          "Action '" + name + "' missing from deep action set.");
    }
    Action tmp = *deep_it;
    // Carry a non-owning pointer back to the grounded plank action. The DEL
    // transition reads everything (preconditions, postconditions, accessibility
    // by obs-type, designated events) directly from this pointee at runtime.
    // Lifetime is bound to Domain's PlankPipeline.
    tmp.set_del_action(&pa);

#ifdef USE_MASTAR
    // mA*-only deep-side classification: action-type → PropositionType,
    // observability-type → fully/partially maps, flat postconditions →
    // EffectsMap. Hard-errors on anything outside the {public-ontic,
    // public-sensing, semi-private-sensing, public-announcement,
    // semi-private-announcement} × {Fully, Partially} envelope.
    const PropositionType ptype =
        action_type_to_proposition(pa.get_action_type_name(), name);
    tmp.set_type(ptype);

    const auto &designated = pa.get_designated_events();
    pl::del::event_id first_designated = 0;
    bool found_designated = false;
    for (auto e : designated) {
      first_designated = e;
      found_designated = true;
      break;
    }
    if (found_designated) {
      const pl::del::formula_ptr &pre = pa.get_precondition(first_designated);
      tmp.add_executability(convert_formula(pre, grounder));
    }

    if (ptype == PropositionType::ONTIC && found_designated) {
      const auto &post = pa.get_postconditions(first_designated);
      for (const auto &[atom, becomes] : post) {
        FluentsSet clause;
        const auto ftype = pl::del::formulas_utils::get_type(becomes);
        if (ftype == pl::del::formula_type::true_formula) {
          clause.insert(atom_to_fluent(atom, grounder));
        } else if (ftype == pl::del::formula_type::false_formula) {
          const std::string &n = m_pipeline.language().get_atom_name(atom);
          clause.insert(grounder.ground_fluent(NEGATION_SYMBOL + n));
        } else {
          ExitHandler::exit_with_message(
              ExitHandler::ExitCode::ParsingError,
              "Action '" + name +
                  "' has a conditional postcondition that is not a constant "
                  "true/false formula; deep's mA* effects model does not "
                  "support this case (rebuild with -DMASTAR=OFF for the "
                  "full-DEL transition).");
        }
        FluentFormula ff;
        ff.insert(clause);
        tmp.add_effect(ff, true_belief_formula());
      }
    }

    for (unsigned long t = 0; t < pa.get_obs_types_number(); ++t) {
      const std::string &t_name = pa.get_obs_type_name(t);
      const bool is_full = (t_name == "Fully");
      const bool is_part = (t_name == "Partially");
      if (!is_full && !is_part) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::ParsingError,
            "Unsupported observability type '" + t_name + "' on action '" +
                name + "'. Supported: Fully, Partially.");
      }
      const auto agents_n = m_pipeline.language().get_agents_number();
      for (pl::del::agent ag = 0; ag < agents_n; ++ag) {
        const pl::del::formula_ptr &cond = pa.get_obs_condition(ag, t);
        const auto cond_type = pl::del::formulas_utils::get_type(cond);
        if (cond_type == pl::del::formula_type::false_formula) {
          continue;
        }
        const Agent deep_ag = agent_to_agent(ag, grounder);
        BeliefFormula deep_cond = convert_formula(cond, grounder);
        if (is_full) {
          tmp.add_fully_observant(deep_ag, deep_cond);
        } else {
          tmp.add_partially_observant(deep_ag, deep_cond);
        }
      }
    }
#else
    // Full-DEL path: nothing else to translate on the deep side. The plank
    // action carries all information needed by compute_successor_del and by
    // KripkeState::is_executable_del. No action-type or observability-type
    // tables are consulted; arbitrary plank action-types and obs-types pass
    // through transparently.
    (void)grounder;
#endif

    actions.erase(deep_it);
    actions.insert(tmp);
  }
}

BeliefFormula
PlankTranslator::convert_formula(const pl::del::formula_ptr &f,
                                 const Grounder &grounder) const {
  // We avoid `using namespace plank::del` here because the EPDDL AST headers
  // (transitively included) declare a parallel set of *_formula_ptr aliases
  // in plank::epddl::ast, which would shadow / collide.
  namespace D = plank::del;
  BeliefFormula bf;

  std::visit(
      [&](auto &&ptr) {
        using T = std::decay_t<decltype(ptr)>;
        if constexpr (std::is_same_v<T, D::true_formula_ptr>) {
          // BF_EMPTY is deep's "vacuously true" sentinel — matches how the
          // old parser produced empty executability/observability conditions.
          bf.set_formula_type(BeliefFormulaType::BF_EMPTY);
        } else if constexpr (std::is_same_v<T, D::false_formula_ptr>) {
          // Encoded as a fluent formula with one impossible clause: empty DNF.
          // Downstream readers should treat empty-clause FF as 'unsatisfiable'.
          bf.set_formula_type(BeliefFormulaType::FLUENT_FORMULA);
          FluentsSet clause;
          // Add fluent + its negation to force unsatisfiability.
          if (m_pipeline.language().get_atoms_number() > 0) {
            const std::string &any = m_pipeline.language().get_atom_name(0);
            clause.insert(grounder.ground_fluent(any));
            clause.insert(grounder.ground_fluent(NEGATION_SYMBOL + any));
          }
          FluentFormula ff;
          ff.insert(clause);
          bf.set_fluent_formula(ff);
        } else if constexpr (std::is_same_v<T, D::atom_formula_ptr>) {
          bf.set_formula_type(BeliefFormulaType::FLUENT_FORMULA);
          FluentsSet clause;
          clause.insert(atom_to_fluent(ptr->get_atom(), grounder));
          FluentFormula ff;
          ff.insert(clause);
          bf.set_fluent_formula(ff);
        } else if constexpr (std::is_same_v<T, D::not_formula_ptr>) {
          // Normalize `not(atom)` into a single-literal FluentFormula with a
          // negative literal, so it satisfies deep's S5 restriction check
          // (PROPOSITIONAL_FORMULA + NOT does not). Anything more complex
          // stays as a NOT-wrapped propositional formula.
          if (auto inner_atom = std::get_if<D::atom_formula_ptr>(
                  &ptr->get_formula())) {
            const std::string &n =
                m_pipeline.language().get_atom_name((*inner_atom)->get_atom());
            FluentsSet clause;
            clause.insert(grounder.ground_fluent(NEGATION_SYMBOL + n));
            FluentFormula ff;
            ff.insert(clause);
            bf.set_formula_type(BeliefFormulaType::FLUENT_FORMULA);
            bf.set_fluent_formula(ff);
          } else {
            bf.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
            bf.set_operator(BeliefFormulaOperator::BF_NOT);
            bf.set_bf1(convert_formula(ptr->get_formula(), grounder));
          }
        } else if constexpr (std::is_same_v<T, D::and_formula_ptr>) {
          // Fold n-ary AND into a left-leaning binary BF_AND tree.
          const auto &subs = ptr->get_formulas();
          if (subs.empty()) {
            bf.set_formula_type(BeliefFormulaType::BF_EMPTY);
          } else if (subs.size() == 1) {
            bf = convert_formula(subs.front(), grounder);
          } else {
            BeliefFormula acc = convert_formula(subs.front(), grounder);
            for (auto it = std::next(subs.begin()); it != subs.end(); ++it) {
              BeliefFormula step;
              step.set_formula_type(
                  BeliefFormulaType::PROPOSITIONAL_FORMULA);
              step.set_operator(BeliefFormulaOperator::BF_AND);
              step.set_bf1(acc);
              step.set_bf2(convert_formula(*it, grounder));
              acc = step;
            }
            bf = acc;
          }
        } else if constexpr (std::is_same_v<T, D::or_formula_ptr>) {
          const auto &subs = ptr->get_formulas();
          // If every disjunct is a (positive or negated) atom, emit a
          // DNF FluentFormula (multi-clause). This keeps the S5 initial-state
          // restriction check happy.
          auto literal_to_fluent =
              [&](const D::formula_ptr &lit) -> std::optional<Fluent> {
            if (auto a = std::get_if<D::atom_formula_ptr>(&lit)) {
              return atom_to_fluent((*a)->get_atom(), grounder);
            }
            if (auto n = std::get_if<D::not_formula_ptr>(&lit)) {
              if (auto inner =
                      std::get_if<D::atom_formula_ptr>(&(*n)->get_formula())) {
                const std::string &name =
                    m_pipeline.language().get_atom_name((*inner)->get_atom());
                return grounder.ground_fluent(NEGATION_SYMBOL + name);
              }
            }
            return std::nullopt;
          };
          bool all_literals = true;
          FluentFormula dnf;
          for (const auto &sub : subs) {
            auto lit = literal_to_fluent(sub);
            if (!lit) { all_literals = false; break; }
            FluentsSet clause;
            clause.insert(*lit);
            dnf.insert(clause);
          }
          if (all_literals && !subs.empty()) {
            bf.set_formula_type(BeliefFormulaType::FLUENT_FORMULA);
            bf.set_fluent_formula(dnf);
            return;
          }
          if (subs.empty()) {
            // Empty OR == false; reuse false encoding.
            bf = convert_formula(
                pl::del::formula_ptr{std::make_shared<D::false_formula>()},
                grounder);
          } else if (subs.size() == 1) {
            bf = convert_formula(subs.front(), grounder);
          } else {
            BeliefFormula acc = convert_formula(subs.front(), grounder);
            for (auto it = std::next(subs.begin()); it != subs.end(); ++it) {
              BeliefFormula step;
              step.set_formula_type(
                  BeliefFormulaType::PROPOSITIONAL_FORMULA);
              step.set_operator(BeliefFormulaOperator::BF_OR);
              step.set_bf1(acc);
              step.set_bf2(convert_formula(*it, grounder));
              acc = step;
            }
            bf = acc;
          }
        } else if constexpr (std::is_same_v<T, D::imply_formula_ptr>) {
          // (A -> B) == (!A or B)
          BeliefFormula not_a;
          not_a.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
          not_a.set_operator(BeliefFormulaOperator::BF_NOT);
          not_a.set_bf1(convert_formula(ptr->get_first_formula(), grounder));
          bf.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
          bf.set_operator(BeliefFormulaOperator::BF_OR);
          bf.set_bf1(not_a);
          bf.set_bf2(convert_formula(ptr->get_second_formula(), grounder));
        } else if constexpr (std::is_same_v<T, D::box_formula_ptr>) {
          // [ag] phi → B(ag, phi). Multi-agent box → E_FORMULA (everybody-knows).
          const auto &ags = ptr->get_mod_index();
          if (ags.size() == 1) {
            bf.set_formula_type(BeliefFormulaType::BELIEF_FORMULA);
            pl::del::agent only = *ags.begin();
            bf.set_agent(agent_to_agent(only, grounder));
            bf.set_bf1(convert_formula(ptr->get_formula(), grounder));
          } else {
            bf.set_formula_type(BeliefFormulaType::E_FORMULA);
            bf.set_group_agents(agent_set_to_agents(ags, grounder));
            bf.set_bf1(convert_formula(ptr->get_formula(), grounder));
          }
        } else if constexpr (std::is_same_v<T, D::kw_box_formula_ptr>) {
          // [Kw. ag] phi  ≡  B(ag, phi) OR B(ag, ¬phi)  (agent knows whether
          // phi). Distinct from [ag] phi: encoding this as plain BELIEF_FORMULA
          // collapses "knows-whether" into "believes", which silently strengthens
          // every goal/precondition that uses Kw. For multi-agent group:
          //   [Kw. G] phi  ≡  E_G(phi) OR E_G(¬phi).
          BeliefFormula inner = convert_formula(ptr->get_formula(), grounder);
          BeliefFormula not_inner;
          not_inner.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
          not_inner.set_operator(BeliefFormulaOperator::BF_NOT);
          not_inner.set_bf1(inner);
          const auto &ags = ptr->get_mod_index();
          auto make_modal = [&](const BeliefFormula &body) {
            BeliefFormula m;
            if (ags.size() == 1) {
              m.set_formula_type(BeliefFormulaType::BELIEF_FORMULA);
              m.set_agent(agent_to_agent(*ags.begin(), grounder));
            } else {
              m.set_formula_type(BeliefFormulaType::E_FORMULA);
              m.set_group_agents(agent_set_to_agents(ags, grounder));
            }
            m.set_bf1(body);
            return m;
          };
          bf.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
          bf.set_operator(BeliefFormulaOperator::BF_OR);
          bf.set_bf1(make_modal(inner));
          bf.set_bf2(make_modal(not_inner));
        } else if constexpr (std::is_same_v<T, D::diamond_formula_ptr>) {
          // <ag> phi  ≡  ¬[ag]¬phi  (ag considers phi possible).
          BeliefFormula inner_not;
          inner_not.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
          inner_not.set_operator(BeliefFormulaOperator::BF_NOT);
          inner_not.set_bf1(convert_formula(ptr->get_formula(), grounder));
          BeliefFormula belief;
          const auto &ags = ptr->get_mod_index();
          if (ags.size() == 1) {
            belief.set_formula_type(BeliefFormulaType::BELIEF_FORMULA);
            pl::del::agent only = *ags.begin();
            belief.set_agent(agent_to_agent(only, grounder));
          } else {
            belief.set_formula_type(BeliefFormulaType::E_FORMULA);
            belief.set_group_agents(agent_set_to_agents(ags, grounder));
          }
          belief.set_bf1(inner_not);
          bf.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
          bf.set_operator(BeliefFormulaOperator::BF_NOT);
          bf.set_bf1(belief);
        } else if constexpr (std::is_same_v<T, D::kw_diamond_formula_ptr>) {
          // <Kw. ag> phi  ≡  ¬[Kw. ag] phi  ≡  ¬B(ag, phi) AND ¬B(ag, ¬phi)
          //   (ag doesn't know whether phi).
          BeliefFormula inner = convert_formula(ptr->get_formula(), grounder);
          BeliefFormula not_inner;
          not_inner.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
          not_inner.set_operator(BeliefFormulaOperator::BF_NOT);
          not_inner.set_bf1(inner);
          const auto &ags = ptr->get_mod_index();
          auto make_modal = [&](const BeliefFormula &body) {
            BeliefFormula m;
            if (ags.size() == 1) {
              m.set_formula_type(BeliefFormulaType::BELIEF_FORMULA);
              m.set_agent(agent_to_agent(*ags.begin(), grounder));
            } else {
              m.set_formula_type(BeliefFormulaType::E_FORMULA);
              m.set_group_agents(agent_set_to_agents(ags, grounder));
            }
            m.set_bf1(body);
            return m;
          };
          auto negate = [](const BeliefFormula &b) {
            BeliefFormula n;
            n.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
            n.set_operator(BeliefFormulaOperator::BF_NOT);
            n.set_bf1(b);
            return n;
          };
          bf.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
          bf.set_operator(BeliefFormulaOperator::BF_AND);
          bf.set_bf1(negate(make_modal(inner)));
          bf.set_bf2(negate(make_modal(not_inner)));
        } else if constexpr (std::is_same_v<T, D::c_box_formula_ptr>) {
          bf.set_formula_type(BeliefFormulaType::C_FORMULA);
          bf.set_group_agents(
              agent_set_to_agents(ptr->get_mod_index(), grounder));
          bf.set_bf1(convert_formula(ptr->get_formula(), grounder));
        } else if constexpr (std::is_same_v<T, D::c_diamond_formula_ptr>) {
          // <C ags> phi == ! C ags ! phi
          BeliefFormula inner_not;
          inner_not.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
          inner_not.set_operator(BeliefFormulaOperator::BF_NOT);
          inner_not.set_bf1(convert_formula(ptr->get_formula(), grounder));
          BeliefFormula c;
          c.set_formula_type(BeliefFormulaType::C_FORMULA);
          c.set_group_agents(
              agent_set_to_agents(ptr->get_mod_index(), grounder));
          c.set_bf1(inner_not);
          bf.set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
          bf.set_operator(BeliefFormulaOperator::BF_NOT);
          bf.set_bf1(c);
        }
      },
      f);

  // Post-process: fold pure boolean combinations of literals into a single
  // FLUENT_FORMULA (DNF). This brings inputs like `C(G, a AND b AND c)` into
  // the canonical `C(G, FF{{a,b,c}})` shape that InitialStateInformation's
  // S5 restriction check accepts. Modal subformulas survive verbatim — only
  // their propositional bodies are collapsed via the recursive convert_formula
  // calls that already happened.
  if (auto collapsed = try_collapse_to_ff(bf)) {
    BeliefFormula out;
    if (collapsed->empty()) {
      out.set_formula_type(BeliefFormulaType::BF_EMPTY);
    } else {
      out.set_formula_type(BeliefFormulaType::FLUENT_FORMULA);
      out.set_fluent_formula(*collapsed);
    }
    return out;
  }
  return bf;
}

void PlankTranslator::build_initial(InitialStateInformation &initial,
                                    const Grounder &grounder) const {
  // 1. (:facts-init ...) — every fact is an immutable fluent asserted in
  //    the pointed condition.
  const auto &spec = m_pipeline.spec();
  const auto &problem = std::get<0>(spec);
  if (problem) {
    for (const auto &item : problem->get_items()) {
      if (std::holds_alternative<pl::epddl::ast::facts_init_ptr>(item)) {
        // The grounder's `info.facts` already holds the grounded fact atom
        // ids. Walk them.
        for (auto fact_atom : m_pipeline.info().facts) {
          FluentsSet clause;
          clause.insert(atom_to_fluent(fact_atom, grounder));
          FluentFormula ff;
          ff.insert(clause);
          initial.add_pointed_condition(ff);
        }
        break;
      }
    }

    // 2. (:init ...) — finitary-S5 theory: each entry becomes either a
    //    pointed condition (if propositional after conversion) or an initial
    //    belief condition.
    for (const auto &item : problem->get_items()) {
      if (!std::holds_alternative<pl::epddl::ast::initial_state_ptr>(item))
        continue;
      const auto &init = std::get<pl::epddl::ast::initial_state_ptr>(item);
      if (!init) break;
      if (!std::holds_alternative<pl::epddl::ast::finitary_S5_theory>(
              init->get_state())) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::ParsingError,
            "Initial state must be expressed as a (:init ...) finitary-S5 "
            "theory; explicit Kripke initial states are not yet supported.");
      }
      const auto &theory = std::get<pl::epddl::ast::finitary_S5_theory>(
          init->get_state());
      // theory is `list<finitary_S5_formula>` =
      //   variant<singleton_list_ptr<..>, and_list_ptr<..>, forall_list_ptr<..>>
      // Recursive walker that flattens singleton + and-list and visits each
      // finitary_S5_formula leaf. forall-lists at the top level are deferred.
      // Build the all-agents set once for ck_formula wrapping.
      AgentsSet all_agents;
      for (auto &[_, ag] : grounder.get_agent_map()) all_agents.insert(ag);

      auto process_formula = [&](const pl::epddl::ast::finitary_S5_formula
                                     &entry) {
        std::visit(
            [&](auto &&p) {
              using P = std::decay_t<decltype(p)>;
              if constexpr (std::is_same_v<P,
                                           pl::epddl::ast::prop_formula_ptr>) {
                pl::del::formula_ptr grounded =
                    pl::epddl::grounder::formulas_and_lists_grounder::
                        build_formula(p->get_formula(), m_pipeline.info());
                BeliefFormula bf = convert_formula(grounded, grounder);
                if (bf.get_formula_type() == BeliefFormulaType::FLUENT_FORMULA)
                  initial.add_pointed_condition(bf.get_fluent_formula());
                else
                  initial.add_initial_condition(bf);
              } else if constexpr (
                  std::is_same_v<P, pl::epddl::ast::ck_formula_ptr>) {
                // `[C.] phi` → C(All, phi)
                pl::del::formula_ptr grounded =
                    pl::epddl::grounder::formulas_and_lists_grounder::
                        build_formula(p->get_formula(), m_pipeline.info());
                BeliefFormula inner = convert_formula(grounded, grounder);
                BeliefFormula wrapped;
                wrapped.set_formula_type(BeliefFormulaType::C_FORMULA);
                wrapped.set_group_agents(all_agents);
                wrapped.set_bf1(inner);
                initial.add_initial_condition(wrapped);
              } else if constexpr (
                  std::is_same_v<P, pl::epddl::ast::ck_k_formula_ptr>) {
                // `[ag] phi` in a finitary-S5 theory is common-knowledge that
                // ag knows phi: C(All, B(ag, phi)). The `ck_` AST prefix
                // signals the implicit C(All, ...) wrapping.
                pl::del::formula_ptr grounded =
                    pl::epddl::grounder::formulas_and_lists_grounder::
                        build_formula(p->get_formula(), m_pipeline.info());
                BeliefFormula inner = convert_formula(grounded, grounder);
                BeliefFormula belief;
                belief.set_formula_type(BeliefFormulaType::BELIEF_FORMULA);
                belief.set_agent(agent_to_agent(
                    pl::epddl::grounder::language_grounder::get_agent_id(
                        p->get_agent(), m_pipeline.info()),
                    grounder));
                belief.set_bf1(inner);
                BeliefFormula wrapped;
                wrapped.set_formula_type(BeliefFormulaType::C_FORMULA);
                wrapped.set_group_agents(all_agents);
                wrapped.set_bf1(belief);
                initial.add_initial_condition(wrapped);
              } else if constexpr (
                  std::is_same_v<P, pl::epddl::ast::ck_kw_formula_ptr> ||
                  std::is_same_v<P, pl::epddl::ast::ck_not_kw_formula_ptr>) {
                // `[Kw. ag] phi`  in a finitary-S5 theory means
                //   C(All, B(ag, phi) OR B(ag, ¬phi))
                // `[¬Kw. ag] phi` means
                //   C(All, ¬B(ag, phi) AND ¬B(ag, ¬phi))
                // The S5 restriction check recognizes these two patterns
                // under C — see check_Bff_notBff in InitialStateInformation.
                pl::del::formula_ptr grounded =
                    pl::epddl::grounder::formulas_and_lists_grounder::
                        build_formula(p->get_formula(), m_pipeline.info());
                BeliefFormula inner = convert_formula(grounded, grounder);
                BeliefFormula not_inner;
                not_inner.set_formula_type(
                    BeliefFormulaType::PROPOSITIONAL_FORMULA);
                not_inner.set_operator(BeliefFormulaOperator::BF_NOT);
                not_inner.set_bf1(inner);

                Agent ag = agent_to_agent(
                    pl::epddl::grounder::language_grounder::get_agent_id(
                        p->get_agent(), m_pipeline.info()),
                    grounder);

                BeliefFormula b_pos;
                b_pos.set_formula_type(BeliefFormulaType::BELIEF_FORMULA);
                b_pos.set_agent(ag);
                b_pos.set_bf1(inner);
                BeliefFormula b_neg;
                b_neg.set_formula_type(BeliefFormulaType::BELIEF_FORMULA);
                b_neg.set_agent(ag);
                b_neg.set_bf1(not_inner);

                BeliefFormula body;
                if constexpr (std::is_same_v<
                                  P, pl::epddl::ast::ck_kw_formula_ptr>) {
                  body.set_formula_type(
                      BeliefFormulaType::PROPOSITIONAL_FORMULA);
                  body.set_operator(BeliefFormulaOperator::BF_OR);
                  body.set_bf1(b_pos);
                  body.set_bf2(b_neg);
                } else {
                  BeliefFormula not_b_pos;
                  not_b_pos.set_formula_type(
                      BeliefFormulaType::PROPOSITIONAL_FORMULA);
                  not_b_pos.set_operator(BeliefFormulaOperator::BF_NOT);
                  not_b_pos.set_bf1(b_pos);
                  BeliefFormula not_b_neg;
                  not_b_neg.set_formula_type(
                      BeliefFormulaType::PROPOSITIONAL_FORMULA);
                  not_b_neg.set_operator(BeliefFormulaOperator::BF_NOT);
                  not_b_neg.set_bf1(b_neg);
                  body.set_formula_type(
                      BeliefFormulaType::PROPOSITIONAL_FORMULA);
                  body.set_operator(BeliefFormulaOperator::BF_AND);
                  body.set_bf1(not_b_pos);
                  body.set_bf2(not_b_neg);
                }
                BeliefFormula wrapped;
                wrapped.set_formula_type(BeliefFormulaType::C_FORMULA);
                wrapped.set_group_agents(all_agents);
                wrapped.set_bf1(body);
                initial.add_initial_condition(wrapped);
              }
            },
            entry);
      };

      std::function<void(const pl::epddl::ast::list<
                         pl::epddl::ast::finitary_S5_formula> &)>
          walk = [&](const pl::epddl::ast::list<
                     pl::epddl::ast::finitary_S5_formula> &lst) {
            std::visit(
                [&](auto &&list_ptr) {
                  using L = std::decay_t<decltype(list_ptr)>;
                  if constexpr (std::is_same_v<
                                    L,
                                    pl::epddl::ast::singleton_list_ptr<
                                        pl::epddl::ast::finitary_S5_formula>>) {
                    process_formula(list_ptr->get_elem());
                  } else if constexpr (
                      std::is_same_v<
                          L, pl::epddl::ast::and_list_ptr<
                                 pl::epddl::ast::finitary_S5_formula>>) {
                    for (const auto &nested : list_ptr->get_list()) {
                      walk(nested);
                    }
                  } else if constexpr (
                      std::is_same_v<
                          L, pl::epddl::ast::forall_list_ptr<
                                 pl::epddl::ast::finitary_S5_formula>>) {
                    // Expand quantified initial conditions by enumerating the
                    // list comprehension's combinations. plank already does
                    // this for ordinary forall_list groundings (see
                    // formulas_and_lists_grounder::build_list specialization);
                    // we reuse its combinations_handler so binding/scope match
                    // exactly. Each binding is pushed onto info.assignment,
                    // the inner list is walked (any formula it grounds will
                    // resolve free variables through that assignment), then
                    // the binding is popped.
                    pl::epddl::grounder::combinations_handler handler{
                        list_ptr->get_list_compr()->get_formal_params(),
                        m_pipeline.info().context,
                        pl::epddl::type_checker::either_type{}};
                    for (const pl::epddl::grounder::combination &c :
                         pl::epddl::grounder::list_comprehensions_handler::all(
                             list_ptr->get_list_compr()->get_condition(),
                             handler, m_pipeline.info())) {
                      m_pipeline.info().assignment.push(
                          handler.get_typed_vars(), c);
                      walk(list_ptr->get_list());
                      m_pipeline.info().assignment.pop();
                    }
                  }
                },
                lst);
          };
      walk(theory);
      break;
    }
  }

  initial.set_ff_forS5();
}

void PlankTranslator::build_goal(FormulaeList &goal,
                                 const Grounder &grounder) const {
  try {
    goal.push_back(convert_formula(m_pipeline.task().goal, grounder));
  } catch (const std::exception &e) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::ParsingError,
        std::string("build_goal threw: ") + e.what());
  }
}
