/**
 * \file InitialStateInformation.cpp
 * \brief Implementation of \ref InitialStateInformation.
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 16, 2025
 */

#include "InitialStateInformation.h"

#include "ArgumentParser.h"
#include "BeliefFormula.h"
#include "utilities/ExitHandler.h"

#include "utilities/FormulaHelper.h"

/// \brief Checks if a belief formula respects the initial restriction.
/// \param[in] to_check The belief formula to check.
/// \return true if the formula respects the restriction, false otherwise.
bool InitialStateInformation::check_restriction(const BeliefFormula &to_check) {
  /* The possible cases are:
   * - *phi* -> all worlds must entail *phi*.
   * - C(B(i,*phi*)) -> all worlds must entail *phi*.
   * - C(B(i,*phi*) \ref BF_OR B(i,-*phi*)) -> only edges conditions.
   * - C(-B(i,*phi*) \ref BeliefFormulaOperator::BF_AND -B(i,-*phi*)) -> only
   * edges conditions.*/
  bool ret = false;
  FluentFormula ff = {}; // Useless
  switch (to_check.get_formula_type()) {
  case BeliefFormulaType::PROPOSITIONAL_FORMULA:
    if (to_check.get_operator() != BeliefFormulaOperator::BF_AND) {
      ret = false;
    } else {
      ret = check_restriction(to_check.get_bf1()) &&
            check_restriction(to_check.get_bf2());
    }
    break;
  case BeliefFormulaType::FLUENT_FORMULA:
    ret = true;
    break;
  case BeliefFormulaType::C_FORMULA: {
    switch (const auto &tmp = to_check.get_bf1(); tmp.get_formula_type()) {
    case BeliefFormulaType::FLUENT_FORMULA:
      ret = true;
      break;
    case BeliefFormulaType::BELIEF_FORMULA:
      ret = (tmp.get_bf1().get_formula_type() ==
             BeliefFormulaType::FLUENT_FORMULA);
      break;
    case BeliefFormulaType::PROPOSITIONAL_FORMULA:
      if (tmp.get_operator() == BeliefFormulaOperator::BF_OR) {
        ret = FormulaHelper::check_Bff_notBff(tmp.get_bf1(), tmp.get_bf2(), ff);
      } else if (tmp.get_operator() == BeliefFormulaOperator::BF_AND) {
        const auto &tmp_nested1 = tmp.get_bf1();
        const auto &tmp_nested2 = tmp.get_bf2();
        if (tmp_nested1.get_formula_type() ==
                BeliefFormulaType::PROPOSITIONAL_FORMULA &&
            tmp_nested2.get_formula_type() ==
                BeliefFormulaType::PROPOSITIONAL_FORMULA &&
            tmp_nested1.get_operator() == BeliefFormulaOperator::BF_NOT &&
            tmp_nested2.get_operator() == BeliefFormulaOperator::BF_NOT) {
          ret = FormulaHelper::check_Bff_notBff(tmp_nested1.get_bf1(),
                                                tmp_nested2.get_bf1(), ff);
        }
      } else {
        ret = false;
      }
      break;
    case BeliefFormulaType::BF_EMPTY:
      ret = true;
      break;
    default:
      ret = false;
      break;
    }
    break;
  }
  case BeliefFormulaType::BF_EMPTY:
    ret = true;
    break;
  default:
    ret = false;
    break;
  }
  if (!ret) {
    auto &os = ArgumentParser::get_instance().get_output_stream();
    os << "[WARNING] The initial condition ";
    to_check.print();
    os << " does not respect the initial state restriction (finitary S5)."
       << std::endl;
  }
  return ret;
}

void InitialStateInformation::add_pointed_condition(
    const FluentFormula &to_add) {
  // Is in DNF form so you have to add these to the fluent of before (all of
  // them)
  m_pointed_world_conditions =
      FormulaHelper::and_ff(m_pointed_world_conditions, to_add);
}

void InitialStateInformation::add_initial_condition(
    const BeliefFormula &to_add) {
  // Top-level AND distributes into separate initial conditions.
  if (to_add.get_formula_type() == BeliefFormulaType::PROPOSITIONAL_FORMULA &&
      to_add.get_operator() == BeliefFormulaOperator::BF_AND) {
    add_initial_condition(to_add.get_bf1());
    add_initial_condition(to_add.get_bf2());
    return;
  }

  // Common-knowledge distributes over conjunction:
  //   C(g, A ∧ B)  ≡  C(g, A) ∧ C(g, B)
  // PlankTranslator's try_collapse_to_ff bails out of DNF distribution above a
  // size cap, leaving large initial constraints like SC's positional-imply
  // bundle as C(g, BF_AND(FF, BF_AND(FF, …))). Splitting them here lets each
  // leaf go through check_restriction as its own C(g, FF), which is in the
  // canonical accepted shape. (We don't distribute OR, since
  // C(g, A ∨ B) ≢ C(g, A) ∨ C(g, B) under common knowledge.)
  if (to_add.get_formula_type() == BeliefFormulaType::C_FORMULA) {
    const BeliefFormula &body = to_add.get_bf1();
    if (body.get_formula_type() == BeliefFormulaType::PROPOSITIONAL_FORMULA &&
        body.get_operator() == BeliefFormulaOperator::BF_AND) {
      BeliefFormula left, right;
      left.set_formula_type(BeliefFormulaType::C_FORMULA);
      left.set_group_agents(to_add.get_group_agents());
      left.set_bf1(body.get_bf1());
      right.set_formula_type(BeliefFormulaType::C_FORMULA);
      right.set_group_agents(to_add.get_group_agents());
      right.set_bf1(body.get_bf2());
      add_initial_condition(left);
      add_initial_condition(right);
      return;
    }
  }

  if (check_restriction(to_add)) {
    m_bf_initial_conditions.push_back(to_add);
    return;
  }

#ifndef USE_MASTAR
  // Under DEL, the Kripke structure comes from plank's grounded del::state
  // (see KripkeState::build_initial_from_plank_state). The S5 restriction
  // check exists only to constrain what deep's naive enumeration path could
  // handle. Partial-ignorance shapes like C(g, ¬B(ag, φ)) are realised in
  // the grounded structure already, so we just store the formula (for the
  // debug bisimulation-equivalence check in FormulaHelper) and skip the
  // hard-exit. Note: we cannot consult Domain::get_instance() here because
  // we are inside Domain's own initialisation, which would recurse.
  m_bf_initial_conditions.push_back(to_add);
  return;
#else
  ExitHandler::exit_with_message(
      ExitHandler::ExitCode::DomainInitialStateRestrictionError,
      "ERROR: The initial state does not respect the required conditions.");
#endif
}

[[nodiscard]] const FluentFormula &
InitialStateInformation::get_pointed_world_conditions() const {
  return m_pointed_world_conditions;
}

[[nodiscard]] const FormulaeList &
InitialStateInformation::get_initial_conditions() const {
  return m_bf_initial_conditions;
}

[[nodiscard]] const FluentFormula &
InitialStateInformation::get_ff_forS5() const {
  return m_ff_forS5;
}

void InitialStateInformation::set_ff_forS5() {
  // The consistency with S5 is already checked
  FluentFormula ret;
  for (const auto &bf : m_bf_initial_conditions) {
    switch (bf.get_formula_type()) {
    case BeliefFormulaType::FLUENT_FORMULA:
      ret = FormulaHelper::and_ff(ret, bf.get_fluent_formula());
      break;
    case BeliefFormulaType::C_FORMULA: {
      switch (const auto tmp = bf.get_bf1(); tmp.get_formula_type()) {
      case BeliefFormulaType::FLUENT_FORMULA: {
        ret = FormulaHelper::and_ff(ret, tmp.get_fluent_formula());
        if (const auto &tmp_ff = tmp.get_fluent_formula(); tmp_ff.size() == 1) {
          const auto &tmp_fs = *tmp_ff.begin();
          for (const auto &f : tmp_fs)
            m_initially_known_fluents.insert(f);
        }
        break;
      }
      case BeliefFormulaType::BELIEF_FORMULA:
        if (tmp.get_bf1().get_formula_type() ==
            BeliefFormulaType::FLUENT_FORMULA) {
          ret = FormulaHelper::and_ff(ret, tmp.get_bf1().get_fluent_formula());
        } else {
          ExitHandler::exit_with_message(
              ExitHandler::ExitCode::DomainInitialStateTypeError,
              "ERROR: Invalid type in initial formulae (FIRST).");
        }
        break;
      case BeliefFormulaType::PROPOSITIONAL_FORMULA:
        break; // Only for edges -- expresses that someone is ignorant - just
               // edges.
      default:
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::DomainInitialStateTypeError,
            "ERROR: Invalid type in initial formulae (SECOND).");
      }
      break;
    }
    default:
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::DomainInitialStateTypeError,
          "ERROR: Invalid type in initial formulae (THIRD).");
    }
  }
  m_ff_forS5 = ret;
}

[[nodiscard]] const FluentsSet &
InitialStateInformation::get_initially_known_fluents() const {
  return m_initially_known_fluents;
}
