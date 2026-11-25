#ifndef __SD_RUNTIME_STEP_BUDGET_H__
#define __SD_RUNTIME_STEP_BUDGET_H__

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/option_types.h"

// Per-step budget planning for sampling: how many evaluations each step needs,
// how many bytes it requires, which weight blocks to evict, and whether the
// step runs whole, segmented, or not at all. This header intentionally depends
// only on the standard library and core/option_types.h so the planner stays
// testable without ggml backends.

namespace sd::step_budget {

    enum Error {
        None = 0,
        InvalidStepCount,
        ScheduleSizeMismatch,
        NonFiniteSigma,
        NegativeSigma,
        ScheduleNotDescending,
        TerminalSigmaNotZero,
        InvalidBudget,
        InvalidSize,
        BudgetExhausted,
        CapacityUnavailable,
        LedgerUnderflow,
        Canceled,
        UnsupportedMethod,
    };

    const char* to_string(Error error);

    enum Admission {
        Evaluate = 0,
        ReclaimThenEvaluate,
        Segmented,
        Rejected,
    };

    const char* to_string(Admission admission);

    struct WeightBlock {
        std::string name;
        size_t bytes            = 0;
        uint64_t last_use_epoch = 0;
        size_t next_use         = 0;  // 0: never used again
        bool requires_demotion  = false;
        bool protected_block    = false;
    };

    class ResidencyProbe {
    public:
        virtual ~ResidencyProbe() = default;

        virtual size_t available_device_bytes()             = 0;
        // Observational only; admission decisions never read this.
        virtual size_t resident_weight_bytes()              = 0;
        virtual std::vector<WeightBlock> evictable_blocks() = 0;
        virtual bool evict(const WeightBlock& block)        = 0;
        virtual bool device_report_wrapped()                = 0;
    };

    struct Options {
        sample_method_t method  = EULER_SAMPLE_METHOD;
        scheduler_t scheduler   = DISCRETE_SCHEDULER;
        std::vector<float> sigmas;
        bool flow_denoiser          = false;
        bool terminal_sigma_is_zero = false;
        size_t compute_bytes        = 0;
        size_t weight_bytes         = 0;
        size_t budget_bytes         = 0;  // 0: no explicit budget
        size_t scratch_reserve_bytes   = 512ull * 1024 * 1024;
        size_t planning_headroom_bytes = 128ull * 1024 * 1024;
        size_t segment_compute_bytes   = 0;  // 0: no segmented fallback
        int preview_interval           = 0;
        bool preview_final_step        = false;
        bool dry_run                   = false;
        const std::atomic<bool>* cancel = nullptr;
    };

    struct Step {
        uint32_t index           = 0;  // 1-based
        float sigma              = 0.0f;
        float sigma_next         = 0.0f;
        uint32_t eval_count      = 0;
        int first_eval_step_code = 0;
        int completion_step_code = 0;
        bool preview             = false;
        size_t required_bytes    = 0;
        size_t usable_bytes      = 0;
        size_t reclaimed_bytes   = 0;
        Admission admission      = Evaluate;
        Error error              = None;
    };

    struct Plan {
        std::vector<Step> steps;
        Error error                 = None;
        size_t total_evaluations    = 0;
        size_t peak_required_bytes  = 0;
        uint32_t rejected_step      = 0;
    };

    // Cross-execution-unit byte ledger. budget 0 means unbounded.
    class DeviceLedger {
    public:
        explicit DeviceLedger(size_t budget_bytes = 0);

        Error reserve(const std::string& owner, size_t bytes);
        Error release(const std::string& owner, size_t bytes);

        size_t committed_bytes() const;
        size_t peak_bytes() const;
        size_t budget_bytes() const;
        size_t remaining_bytes() const;  // SIZE_MAX when unbounded
        bool fits(size_t bytes) const;

    private:
        mutable std::mutex mutex_;
        size_t budget_bytes_;
        size_t committed_ = 0;
        size_t peak_      = 0;
        std::unordered_map<std::string, size_t> owner_bytes_;
    };

    // Pure planning: reads the probe and the ledger's remaining bytes, never
    // reserves or releases. The same inputs always produce the same Plan.
    Plan plan(const Options& options, ResidencyProbe& probe, DeviceLedger* ledger);

}  // namespace sd::step_budget

#endif  // __SD_RUNTIME_STEP_BUDGET_H__
