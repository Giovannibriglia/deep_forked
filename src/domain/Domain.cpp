/**
 * \brief Implementation of \ref Domain.h.
 *
 * Domain construction is now driven by plank's EPDDL pipeline:
 *   1. Build PlankPipeline   (parse + type-check + ground via plank libs).
 *   2. PlankTranslator copies the grounded data into deep's runtime objects
 *      (agents, fluents, actions, initial description, goal description).
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 14, 2025
 */
#include "Domain.h"

#include <filesystem>

#include "ArgumentParser.h"
#include "Configuration.h"
#include "ExitHandler.h"
#include "HelperPrint.h"
#include "parse/PlankPipeline.h"
#include "parse/PlankTranslator.h"

Domain::Domain() {
  const auto &args = ArgumentParser::get_instance();
  const std::filesystem::path problem_path(args.get_problem_path());
  // Use the problem file's stem as the domain identifier; fall back to its
  // parent directory when the stem is a generic split-name like Test/Training.
  std::string stem = problem_path.stem().string();
  if (stem == "Test" || stem == "Training") {
    m_name = problem_path.parent_path().stem().string();
  } else {
    m_name = stem;
  }

  build();
}

Domain::~Domain() = default;

Domain &Domain::get_instance() {
  static Domain instance;
  return instance;
}

const FluentsSet &Domain::get_fluents() const noexcept { return m_fluents; }

const std::vector<Fluent> &Domain::get_positive_fluents() const noexcept {
  return m_positive_fluents;
}

unsigned int Domain::get_fluent_number() const noexcept {
  return static_cast<unsigned int>(m_fluents.size() / 2);
}

unsigned int Domain::get_size_fluent() const noexcept {
  auto fluent_first = m_fluents.begin();
  return fluent_first != m_fluents.end()
             ? static_cast<unsigned int>(fluent_first->size())
             : 0;
}

const ActionsSet &Domain::get_actions() const noexcept { return m_actions; }

const AgentsSet &Domain::get_agents() const noexcept { return m_agents; }

unsigned int Domain::get_agent_number() const noexcept {
  return static_cast<unsigned int>(m_agents.size());
}

const std::string &Domain::get_name() const noexcept { return m_name; }

const InitialStateInformation &
Domain::get_initial_description() const noexcept {
  return m_initial_description;
}

const FormulaeList &Domain::get_goal_description() const noexcept {
  return m_goal_description;
}

void Domain::build() {
  const auto &args = ArgumentParser::get_instance();
  auto &os = args.get_output_stream();
  const bool verbose = args.get_verbose();

  if (verbose) {
    os << "\n\n========== DOMAIN OUTPUT BEGIN ==========\n";
  }

  m_pipeline = std::make_unique<PlankPipeline>(
      args.get_problem_path(), args.get_domain_path(),
      args.get_libraries_paths(), verbose);
  PlankTranslator translator(*m_pipeline);

  Grounder grounder;

  if (verbose) os << "Building agent list..." << std::endl;
  AgentsMap agent_map;
  translator.build_agents(m_agents, agent_map);
  grounder.set_agent_map(agent_map);

  if (verbose) os << "Building fluent literals..." << std::endl;
  FluentMap fluent_map;
  translator.build_fluents(m_fluents, m_positive_fluents, fluent_map);
  grounder.set_fluent_map(fluent_map);

  if (verbose) os << "Building action list..." << std::endl;
  ActionNamesMap action_name_map;
  translator.build_actions(m_actions, action_name_map);
  grounder.set_action_name_map(action_name_map);

  HelperPrint::get_instance().set_grounder(grounder);

  if (verbose) os << "Populating action effects/observability..." << std::endl;
  translator.populate_actions(m_actions, grounder);

  if (verbose) {
    os << "\nPrinting complete action list..." << std::endl;
    for (const auto &action : m_actions) {
      action.print();
    }
  }

  if (verbose) os << "Adding to pointed world and initial conditions..."
                  << std::endl;
  translator.build_initial(m_initial_description, grounder);

  if (verbose) os << "Adding to Goal..." << std::endl;
  translator.build_goal(m_goal_description, grounder);

  if (verbose) {
    os << "========== DOMAIN OUTPUT END ==========\n\n";
  }
}
