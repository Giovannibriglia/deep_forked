#include "PlankPipeline.h"

#include "ArgumentParser.h"
#include "utilities/ExitHandler.h"

#include "epddl/error-manager/epddl_exception.h"
#include "epddl/grounder/grounder_helper.h"
#include "epddl/parser/file_parser.h"
#include "epddl/type-checker/type_checker.h"

PlankPipeline::PlankPipeline(const std::string &problem_path,
                             const std::string &domain_path,
                             const std::vector<std::string> &libraries_paths,
                             bool verbose) {
  auto &os = ArgumentParser::get_instance().get_output_stream();

  plank::epddl::parser::specification_paths spec_paths{problem_path, domain_path,
                                                       libraries_paths};

  try {
    if (verbose) {
      os << "Parsing..." << std::flush;
    }
    auto [spec, err_managers] =
        plank::epddl::parser::file_parser::parse_planning_specification(
            spec_paths);
    if (verbose) {
      os << " done." << std::endl;
      os << "Type-checking..." << std::flush;
    }
    plank::epddl::type_checker::context context =
        plank::epddl::type_checker::do_semantic_check(spec, err_managers);
    if (verbose) {
      os << " done." << std::endl;
      os << "Grounding..." << std::flush;
    }
    auto [task, info] = plank::epddl::grounder::grounder_helper::ground(
        spec, context, err_managers);
    if (verbose) {
      os << " done." << std::endl;
    }

    m_spec = std::make_unique<plank::epddl::ast::planning_specification>(
        std::move(spec));
    m_context =
        std::make_unique<plank::epddl::type_checker::context>(std::move(context));
    m_info = std::make_unique<plank::epddl::grounder::grounder_info>(
        std::move(info));
    m_task = std::make_unique<plank::del::planning_task>(std::move(task));
  } catch (plank::epddl::EPDDLException &e) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::ParsingError,
        std::string("EPDDL parser error: ") + e.what());
  }
}
