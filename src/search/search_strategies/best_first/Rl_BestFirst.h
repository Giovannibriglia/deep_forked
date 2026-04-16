#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "BestFirst.h"

enum class RefillMode {
  RANDOM,
  HEURISTIC
};

template<StateRepresentation StateRepr>
class RL_BestFirst final : public BestFirst<StateRepr> {
public:
  using Base = BestFirst<StateRepr>;
  using Base::Base;

  explicit RL_BestFirst(const State<StateRepr> &initial_state)
    : Base(initial_state) {
    m_refill_mode = ArgumentParser::get_instance().get_rl_refill_mode() == "random"
                      ? RefillMode::RANDOM
                      : RefillMode::HEURISTIC;

    m_seed = ArgumentParser::get_instance().get_rl_beam_seed();
    if (m_seed < 0) {
      m_seed = static_cast<int64_t>(std::random_device{}());
    }
    m_rng.seed(m_seed);
  }

  void set_refill_mode(const RefillMode mode) {
    m_refill_mode = mode;
    if (m_refill_mode == RefillMode::HEURISTIC && !m_reservoir.empty()) {
      std::make_heap(m_reservoir.begin(), m_reservoir.end(), ReservoirCompare{});
    }
  }

  void push([[maybe_unused]] State<StateRepr> &s) override {
    ExitHandler::exit_with_message(
      ExitHandler::ExitCode::SearchMethodNotImplemented,
      "Error: push of a single state is not implemented for RL_BestFirst.");
  }

  void push(const std::vector<State<StateRepr> > &states) override {
    if (states.empty() && m_reservoir.empty()) {
      return;
    }

    std::vector<State<StateRepr> > batch;
    batch.reserve(m_max_beam_size);

    // Add the newly generated states first.
    // If there are more than m_max_beam_size, the extra states go directly to the reservoir.
    for (const auto &s: states) {
      if (batch.size() < m_max_beam_size) {
        batch.push_back(s);
      } else {
        reservoir_push(State<StateRepr>(s), 0);
      }
    }

    // Fill the remaining slots from the reservoir before heuristic evaluation.
    refill_beam(batch);

    if (batch.empty()) {
      return;
    }

    const std::vector<int> heuristic_values =
        this->m_heuristics_manager.get_heuristic_value(batch);

    for (std::size_t i = 0; i < batch.size(); ++i) {
      batch[i].set_heuristic_value(heuristic_values[i]);
    }

    // Keep the best state in the active queue; return the rest to the reservoir.
    const auto best_it = std::min_element(
      batch.begin(), batch.end(),
      [](const State<StateRepr> &lhs, const State<StateRepr> &rhs) {
        if (lhs.get_heuristic_value() != rhs.get_heuristic_value()) {
          return lhs.get_heuristic_value() < rhs.get_heuristic_value();
        }
        return lhs < rhs;
      });

    if (best_it != batch.end()) {
      this->search_space.push(*best_it);
    }
    else {
      ExitHandler::exit_with_message(
        ExitHandler::ExitCode::SearchMethodNotImplemented,
        "Error: No valid states to push into the search space.");
    }

    for (auto it = batch.begin(); it != batch.end(); ++it) {
      if (it != best_it) {
        reservoir_push(std::move(*it), it->get_heuristic_value());
      }
    }
  }

  void reset() override {
    Base::reset();
    m_reservoir.clear();
    m_rng.seed(m_seed);
  }

  [[nodiscard]] std::string get_name() const override {
    return "RLBeam x BestFirst search (" + this->m_heuristics_manager.get_used_h_name() +
           ", " + (m_refill_mode == RefillMode::RANDOM ? "random" : "heuristic") + ")";
  }

private:
  std::size_t m_max_beam_size = ArgumentParser::get_instance().get_max_fringe_size();
  RefillMode m_refill_mode{RefillMode::HEURISTIC};

  std::vector<State<StateRepr> > m_reservoir;
  std::size_t m_exploration_size = 2;

  std::mt19937_64 m_rng;
  int64_t m_seed{-1};

  struct ReservoirCompare {
    bool operator()(const State<StateRepr> &lhs, const State<StateRepr> &rhs) const {
      const int lh = lhs.get_heuristic_value();
      const int rh = rhs.get_heuristic_value();
      if (lh != rh) {
        return lh > rh; // smaller heuristic = higher priority
      }
      return rhs < lhs;
    }
  };

  void refill_beam(std::vector<State<StateRepr> > &batch) {
    if (batch.size() >= m_max_beam_size || m_reservoir.empty()) {
      return;
    }

    if (m_refill_mode == RefillMode::RANDOM) {
      refill_beam_random(batch);
    } else {
      refill_beam_heuristic(batch);
    }
  }

  void refill_beam_random(std::vector<State<StateRepr> > &batch) {
    while (batch.size() < m_max_beam_size && !m_reservoir.empty()) {
      batch.push_back(reservoir_take_random());
    }
  }

  void refill_beam_heuristic(std::vector<State<StateRepr> > &batch) {
    const std::size_t free_slots = m_max_beam_size - batch.size();
    if (free_slots == 0 || m_reservoir.empty()) {
      return;
    }

    // Fill most of the missing slots with the best reservoir states.
    // Keep a small amount of random exploration inside the beam budget.
    const std::size_t exploration_slots =
        std::min<std::size_t>({free_slots, m_reservoir.size(), m_exploration_size});
    const std::size_t exploit_slots = free_slots - exploration_slots;

    for (std::size_t i = 0; i < exploit_slots && !m_reservoir.empty(); ++i) {
      batch.push_back(reservoir_take_best());
    }

    for (std::size_t i = 0; i < exploration_slots && !m_reservoir.empty(); ++i) {
      batch.push_back(reservoir_take_random());
    }

    if (exploration_slots > 0 && !m_reservoir.empty()) {
      std::make_heap(m_reservoir.begin(), m_reservoir.end(), ReservoirCompare{});
    }
  }

  void reservoir_push(State<StateRepr> &&candidate, int heuristic_value) {
    //Change Heuristic value to be a different one maybe (like number of subgoals)
    candidate.set_heuristic_value(heuristic_value);

    if (m_refill_mode == RefillMode::RANDOM) {
      m_reservoir.push_back(std::move(candidate));
      return;
    }

    m_reservoir.push_back(std::move(candidate));
    std::push_heap(m_reservoir.begin(), m_reservoir.end(), ReservoirCompare{});
  }

  State<StateRepr> reservoir_take_random() {
    std::uniform_int_distribution<std::size_t> distribution(0, m_reservoir.size() - 1);
    const std::size_t idx = distribution(m_rng);

    State<StateRepr> candidate = std::move(m_reservoir[idx]);
    m_reservoir[idx] = std::move(m_reservoir.back());
    m_reservoir.pop_back();
    return candidate;
  }

  State<StateRepr> reservoir_take_best() {
    std::pop_heap(m_reservoir.begin(), m_reservoir.end(), ReservoirCompare{});
    State<StateRepr> candidate = std::move(m_reservoir.back());
    m_reservoir.pop_back();
    return candidate;
  }

  void pop() override {
    if (this->search_space.empty()) {
      if (m_reservoir.empty()) {
        return;
      }

      if (m_refill_mode == RefillMode::RANDOM) {
        this->search_space.push(reservoir_take_random());
      } else {
        this->search_space.push(reservoir_take_best());
      }
      return;
    }

    this->search_space.pop();
  }
};
