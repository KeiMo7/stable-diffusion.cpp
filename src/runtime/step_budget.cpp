#include "runtime/step_budget.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "runtime/preview_interval.h"

namespace sd::step_budget {

    const char* to_string(Error error) {
        switch (error) {
            case None: return "none";
            case InvalidStepCount: return "invalid_step_count";
            case ScheduleSizeMismatch: return "schedule_size_mismatch";
            case NonFiniteSigma: return "non_finite_sigma";
            case NegativeSigma: return "negative_sigma";
            case ScheduleNotDescending: return "schedule_not_descending";
            case TerminalSigmaNotZero: return "terminal_sigma_not_zero";
            case InvalidBudget: return "invalid_budget";
            case InvalidSize: return "invalid_size";
            case BudgetExhausted: return "budget_exhausted";
            case CapacityUnavailable: return "capacity_unavailable";
            case LedgerUnderflow: return "ledger_underflow";
            case Canceled: return "canceled";
            case UnsupportedMethod: return "unsupported_method";
        }
        return "unknown";
    }

    const char* to_string(Admission admission) {
        switch (admission) {
            case Evaluate: return "evaluate";
            case ReclaimThenEvaluate: return "reclaim_then_evaluate";
            case Segmented: return "segmented";
            case Rejected: return "rejected";
        }
        return "unknown";
    }

    DeviceLedger::DeviceLedger(size_t budget_bytes)
        : budget_bytes_(budget_bytes) {}

    Error DeviceLedger::reserve(const std::string& owner, size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (bytes > SIZE_MAX - committed_) {
            return InvalidSize;
        }
        if (budget_bytes_ != 0 && committed_ + bytes > budget_bytes_) {
            return BudgetExhausted;
        }
        committed_ += bytes;
        owner_bytes_[owner] += bytes;
        if (committed_ > peak_) {
            peak_ = committed_;
        }
        return None;
    }

    Error DeviceLedger::release(const std::string& owner, size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = owner_bytes_.find(owner);
        if (it == owner_bytes_.end() || it->second < bytes) {
            return LedgerUnderflow;
        }
        it->second -= bytes;
        committed_ -= bytes;
        return None;
    }

    size_t DeviceLedger::committed_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return committed_;
    }

    size_t DeviceLedger::peak_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return peak_;
    }

    size_t DeviceLedger::budget_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return budget_bytes_;
    }

    size_t DeviceLedger::remaining_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (budget_bytes_ == 0) {
            return SIZE_MAX;
        }
        return budget_bytes_ - committed_;
    }

    bool DeviceLedger::fits(size_t bytes) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (budget_bytes_ == 0) {
            return true;
        }
        return bytes <= budget_bytes_ - committed_;
    }

    namespace {

        size_t sat_add(size_t a, size_t b) {
            return b > SIZE_MAX - a ? SIZE_MAX : a + b;
        }

        size_t sat_sub(size_t a, size_t b) {
            return b > a ? 0 : a - b;
        }

        // Deterministic eviction order: demotion candidates last; within each
        // group by next_use (0 = never used again, first), last_use_epoch,
        // bytes descending, then name. No container iteration order leaks in.
        bool weight_block_order(const WeightBlock& lhs, const WeightBlock& rhs) {
            if (lhs.requires_demotion != rhs.requires_demotion) {
                return !lhs.requires_demotion;
            }
            if (lhs.next_use != rhs.next_use) {
                return lhs.next_use < rhs.next_use;
            }
            if (lhs.last_use_epoch != rhs.last_use_epoch) {
                return lhs.last_use_epoch < rhs.last_use_epoch;
            }
            if (lhs.bytes != rhs.bytes) {
                return lhs.bytes > rhs.bytes;
            }
            return lhs.name < rhs.name;
        }

        uint32_t step_eval_count(const Options& options, uint32_t index, float sigma, float sigma_next) {
            switch (options.method) {
                case EULER_SAMPLE_METHOD:
                    return sigma == sigma_next ? 0u : 1u;
                case HEUN_SAMPLE_METHOD:
                case DPM2_SAMPLE_METHOD:
                    return sigma_next > 0.0f ? 2u : 1u;
                case DPMPP2S_A_SAMPLE_METHOD:
                    if (options.flow_denoiser && index == 1 && 1.0f - sigma < 1e-6f) {
                        return 1u;
                    }
                    return sigma_next > 0.0f ? 2u : 1u;
                default:
                    return 1u;
            }
        }

        bool cancel_requested(const Options& options) {
            return options.cancel != nullptr && options.cancel->load(std::memory_order_relaxed);
        }

        void log_step_decision(const Step& step) {
            std::fprintf(stderr,
                         "step-budget: step=%u eval=%u required=%zu usable=%zu reclaimed=%zu admission=%d\n",
                         step.index,
                         step.eval_count,
                         step.required_bytes,
                         step.usable_bytes,
                         step.reclaimed_bytes,
                         static_cast<int>(step.admission));
        }

    }  // namespace

    Plan plan(const Options& options, ResidencyProbe& probe, DeviceLedger* ledger) {
        Plan result;

        const size_t sigma_count = options.sigmas.size();
        if (sigma_count < 2) {
            result.error = InvalidStepCount;
            return result;
        }
        const size_t num_steps = sigma_count - 1;
        if (options.scheduler == GITS_SCHEDULER && num_steps < 2) {
            result.error = InvalidStepCount;
            return result;
        }
        if (static_cast<int>(options.method) < 0 ||
            static_cast<int>(options.method) >= static_cast<int>(SAMPLE_METHOD_COUNT)) {
            result.error = UnsupportedMethod;
            return result;
        }
        for (float sigma : options.sigmas) {
            if (!std::isfinite(sigma)) {
                result.error = NonFiniteSigma;
                return result;
            }
        }
        if (options.sigmas.front() <= 0.0f) {
            result.error = NegativeSigma;
            return result;
        }
        for (size_t i = 0; i + 1 < sigma_count; i++) {
            if (options.sigmas[i] < options.sigmas[i + 1]) {
                result.error = ScheduleNotDescending;
                return result;
            }
        }
        if (options.terminal_sigma_is_zero && options.sigmas.back() != 0.0f) {
            result.error = TerminalSigmaNotZero;
            return result;
        }
        const size_t reserve_total = sat_add(options.scratch_reserve_bytes, options.planning_headroom_bytes);
        const bool has_budget      = options.budget_bytes != 0;
        if (has_budget && options.budget_bytes <= reserve_total) {
            result.error = InvalidBudget;
            return result;
        }

        const bool wrapped = probe.device_report_wrapped();
        if (wrapped && !has_budget) {
            result.error = CapacityUnavailable;
            return result;
        }

        // Eviction candidates are fetched lazily on the first step that needs
        // reclamation, sorted once, and consumed in order: evicted blocks stay
        // gone for the rest of the plan, so later steps continue down the same
        // sorted sequence. Indices are sorted instead of blocks so no weight
        // metadata is copied.
        bool candidates_loaded = false;
        std::vector<WeightBlock> blocks;
        std::vector<uint32_t> candidate_order;
        size_t next_candidate = 0;

        for (size_t i = 0; i < num_steps; i++) {
            if (cancel_requested(options)) {
                result.error = Canceled;
                break;
            }

            Step step;
            step.index      = static_cast<uint32_t>(i + 1);
            step.sigma      = options.sigmas[i];
            step.sigma_next = options.sigmas[i + 1];
            step.eval_count = step_eval_count(options, step.index, step.sigma, step.sigma_next);
            if (step.eval_count == 2) {
                step.first_eval_step_code = -static_cast<int>(step.index);
                step.completion_step_code = static_cast<int>(step.index);
            } else if (step.eval_count == 1) {
                step.first_eval_step_code = static_cast<int>(step.index);
                step.completion_step_code = static_cast<int>(step.index);
            }
            step.preview = step.eval_count > 0 &&
                           sd::preview::should_preview_sample_step(step.completion_step_code,
                                                                   num_steps,
                                                                   options.terminal_sigma_is_zero,
                                                                   options.preview_interval,
                                                                   options.preview_final_step);
            step.required_bytes = sat_add(options.compute_bytes, options.weight_bytes);

            size_t head = 0;
            if (wrapped) {
                // Wrapped device report with an explicit budget: the free
                // reading cannot be trusted, plan from the budget alone.
                head = ledger != nullptr ? ledger->remaining_bytes() : options.budget_bytes;
            } else {
                head = probe.available_device_bytes();
                if (has_budget) {
                    const size_t remaining =
                        ledger != nullptr ? ledger->remaining_bytes() : options.budget_bytes;
                    head = std::min(head, remaining);
                }
            }
            step.usable_bytes = sat_sub(head, reserve_total);

            if (step.required_bytes > step.usable_bytes) {
                const size_t segment_need = options.segment_compute_bytes != 0
                                                ? sat_add(options.segment_compute_bytes, options.weight_bytes)
                                                : 0;
                const size_t target = options.segment_compute_bytes != 0
                                          ? std::min(step.required_bytes, segment_need)
                                          : step.required_bytes;

                if (!candidates_loaded) {
                    candidates_loaded = true;
                    blocks            = probe.evictable_blocks();
                    candidate_order.reserve(blocks.size());
                    for (size_t b = 0; b < blocks.size(); b++) {
                        if (!blocks[b].protected_block) {
                            candidate_order.push_back(static_cast<uint32_t>(b));
                        }
                    }
                    std::sort(candidate_order.begin(), candidate_order.end(),
                              [&blocks](uint32_t lhs, uint32_t rhs) {
                                  return weight_block_order(blocks[lhs], blocks[rhs]);
                              });
                }

                while (next_candidate < candidate_order.size() &&
                       sat_add(step.usable_bytes, step.reclaimed_bytes) < target) {
                    const WeightBlock& block = blocks[candidate_order[next_candidate]];
                    ++next_candidate;
                    if (!options.dry_run && !probe.evict(block)) {
                        // Eviction failure aborts planning; recorded steps stay.
                        result.error = CapacityUnavailable;
                        return result;
                    }
                    step.reclaimed_bytes = sat_add(step.reclaimed_bytes, block.bytes);
                }

                const size_t covered = sat_add(step.usable_bytes, step.reclaimed_bytes);
                if (covered >= step.required_bytes) {
                    step.admission = ReclaimThenEvaluate;
                } else if (options.segment_compute_bytes != 0 && covered >= segment_need) {
                    step.admission = Segmented;
                } else {
                    step.admission = Rejected;
                    step.error     = BudgetExhausted;
                }
            }

            if (step.admission == Segmented || step.admission == Rejected) {
                log_step_decision(step);
            }

            result.peak_required_bytes = std::max(result.peak_required_bytes, step.required_bytes);
            result.total_evaluations += step.eval_count;
            result.steps.push_back(step);

            if (step.admission == Rejected) {
                result.error         = BudgetExhausted;
                result.rejected_step = step.index;
                break;
            }
            if (cancel_requested(options)) {
                result.error = Canceled;
                break;
            }
        }

        return result;
    }

}  // namespace sd::step_budget
