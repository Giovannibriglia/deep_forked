/**
 * \class PlankPipeline
 * \brief Owns the end-to-end EPDDL parsing + type-checking + grounding pipeline.
 *
 * Replaces the old Flex/Bison `Reader::read()` entry point. Drives plank's
 *   parse  -> type-check  -> ground
 * stages and exposes the resulting AST, grounder info, and grounded
 * `del::planning_task` for downstream consumption by `PlankTranslator`.
 *
 * The class is non-copyable; everything inside (AST, language, planning task)
 * is owned by `std::unique_ptr` because plank's `grounder_info` carries a
 * `const language_ptr` member and the AST's `act_type_library_ptr`s are not
 * copyable as a list.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "epddl/ast/planning_specification.h"
#include "epddl/grounder/grounder_info.h"
#include "epddl/type-checker/context/context.h"
#include "del/semantics/planning_task.h"

class PlankPipeline {
public:
  PlankPipeline(const std::string &problem_path,
                const std::string &domain_path,
                const std::vector<std::string> &libraries_paths, bool verbose);

  PlankPipeline(const PlankPipeline &) = delete;
  PlankPipeline &operator=(const PlankPipeline &) = delete;

  [[nodiscard]] const plank::epddl::ast::planning_specification &
  spec() const noexcept {
    return *m_spec;
  }
  [[nodiscard]] const plank::epddl::grounder::grounder_info &
  info() const noexcept {
    return *m_info;
  }
  // Non-const access is needed because plank's formula-grounding helpers take
  // `grounder_info &` (they mutate the variables_assignment internally during
  // quantifier expansion).
  [[nodiscard]] plank::epddl::grounder::grounder_info &info() noexcept {
    return *m_info;
  }
  [[nodiscard]] const plank::del::planning_task &task() const noexcept {
    return *m_task;
  }
  [[nodiscard]] const plank::del::language &language() const noexcept {
    return *m_info->language;
  }

private:
  std::unique_ptr<plank::epddl::ast::planning_specification> m_spec;
  std::unique_ptr<plank::epddl::type_checker::context> m_context;
  std::unique_ptr<plank::epddl::grounder::grounder_info> m_info;
  std::unique_ptr<plank::del::planning_task> m_task;
};
