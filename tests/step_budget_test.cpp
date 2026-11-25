#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "runtime/step_budget.h"

#ifdef _WIN32
#include <io.h>
#define sd_dup _dup
#define sd_dup2 _dup2
#define sd_close _close
#define sd_fileno _fileno
#else
#include <unistd.h>
#define sd_dup dup
#define sd_dup2 dup2
#define sd_close close
#define sd_fileno fileno
#endif

using sd::step_budget::Admission;
using sd::step_budget::DeviceLedger;
using sd::step_budget::Error;
using sd::step_budget::Options;
using sd::step_budget::Plan;
using sd::step_budget::Step;
using sd::step_budget::WeightBlock;

static int g_checks   = 0;
static int g_failures = 0;

#define CHECK_TRUE(cond)                                        \
    do {                                                        \
        ++g_checks;                                             \
        if (!(cond)) {                                          \
            ++g_failures;                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__,  \
                         __LINE__, #cond);                      \
        }                                                       \
    } while (0)

#define CHECK_EQ(a, b)                                                          \
    do {                                                                        \
        ++g_checks;                                                             \
        long long sd_a = static_cast<long long>(a);                             \
        long long sd_b = static_cast<long long>(b);                             \
        if (sd_a != sd_b) {                                                     \
            ++g_failures;                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s == %s (%lld != %lld)\n",       \
                         __FILE__, __LINE__, #a, #b, sd_a, sd_b);               \
        }                                                                       \
    } while (0)

#define CHECK_STR(a, b)                                                         \
    do {                                                                        \
        ++g_checks;                                                             \
        std::string sd_a = (a);                                                 \
        std::string sd_b = (b);                                                 \
        if (sd_a != sd_b) {                                                     \
            ++g_failures;                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s == %s (%s != %s)\n",           \
                         __FILE__, __LINE__, #a, #b, sd_a.c_str(),              \
                         sd_b.c_str());                                         \
        }                                                                       \
    } while (0)

struct MockProbe : sd::step_budget::ResidencyProbe {
    size_t available   = 0;
    bool wrapped       = false;
    bool evict_result  = true;
    size_t evict_calls = 0;
    size_t available_calls = 0;
    bool evict_frees_memory = false;
    std::vector<WeightBlock> blocks;
    std::vector<std::string> evicted;
    std::function<void(MockProbe&, size_t)> on_available;

    size_t available_device_bytes() override {
        ++available_calls;
        size_t value = available;
        if (on_available) {
            on_available(*this, available_calls);
        }
        return value;
    }

    size_t resident_weight_bytes() override {
        size_t total = 0;
        for (const WeightBlock& block : blocks) {
            total += block.bytes;
        }
        return total;
    }

    std::vector<WeightBlock> evictable_blocks() override { return blocks; }

    bool evict(const WeightBlock& block) override {
        ++evict_calls;
        if (evict_result) {
            evicted.push_back(block.name);
            if (evict_frees_memory) {
                available += block.bytes;
            }
        }
        return evict_result;
    }

    bool device_report_wrapped() override { return wrapped; }
};

static WeightBlock make_block(const std::string& name,
                              size_t bytes,
                              size_t next_use        = 0,
                              uint64_t last_use_epoch = 0,
                              bool requires_demotion  = false,
                              bool protected_block    = false) {
    WeightBlock block;
    block.name              = name;
    block.bytes             = bytes;
    block.next_use          = next_use;
    block.last_use_epoch    = last_use_epoch;
    block.requires_demotion = requires_demotion;
    block.protected_block   = protected_block;
    return block;
}

// required_bytes = 1500, reserve total = 150.
static Options base_options() {
    Options o;
    o.method                   = EULER_SAMPLE_METHOD;
    o.scheduler                = DISCRETE_SCHEDULER;
    o.terminal_sigma_is_zero   = true;
    o.compute_bytes            = 1000;
    o.weight_bytes             = 500;
    o.scratch_reserve_bytes    = 100;
    o.planning_headroom_bytes  = 50;
    return o;
}

static std::vector<float> countdown_sigmas(int start, int count) {
    std::vector<float> sigmas;
    for (int i = 0; i < count; i++) {
        sigmas.push_back(static_cast<float>(start - i));
    }
    return sigmas;
}

static void test_to_string() {
    CHECK_STR(sd::step_budget::to_string(sd::step_budget::None), "none");
    CHECK_STR(sd::step_budget::to_string(sd::step_budget::InvalidStepCount), "invalid_step_count");
    CHECK_STR(sd::step_budget::to_string(sd::step_budget::ScheduleSizeMismatch), "schedule_size_mismatch");
    CHECK_STR(sd::step_budget::to_string(sd::step_budget::NonFiniteSigma), "non_finite_sigma");
    CHECK_STR(sd::step_budget::to_string(sd::step_budget::NegativeSigma), "negative_sigma");
    CHECK_STR(sd::step_budget::to_string(sd::step_budget::ScheduleNotDescending), "schedule_not_descending");
    CHECK_STR(sd::step_budget::to_string(sd::step_budget::TerminalSigmaNotZero), "terminal_sigma_not_zero");
    CHECK_STR(sd::step_budget::to_string(sd::step_budget::InvalidBudget), "invalid_budget");
    CHECK_STR(sd::step_budget::to_string(sd::step_budget::InvalidSize), "invalid_size");
    CHECK_STR(sd::step_budget::to_string(sd::step_budget::BudgetExhausted), "budget_exhausted");
    CHECK_STR(sd::step_budget::to_string(sd::step_budget::CapacityUnavailable), "capacity_unavailable");
    CHECK_STR(sd::step_budget::to_string(sd::step_budget::LedgerUnderflow), "ledger_underflow");
    CHECK_STR(sd::step_budget::to_string(sd::step_budget::Canceled), "canceled");
    CHECK_STR(sd::step_budget::to_string(sd::step_budget::UnsupportedMethod), "unsupported_method");

    CHECK_STR(sd::step_budget::to_string(sd::step_budget::Evaluate), "evaluate");
    CHECK_STR(sd::step_budget::to_string(sd::step_budget::ReclaimThenEvaluate), "reclaim_then_evaluate");
    CHECK_STR(sd::step_budget::to_string(sd::step_budget::Segmented), "segmented");
    CHECK_STR(sd::step_budget::to_string(sd::step_budget::Rejected), "rejected");
}

static void test_eval_counts() {
    MockProbe probe;
    probe.available = 1ull << 40;

    Options o  = base_options();
    o.sigmas   = countdown_sigmas(20, 21);  // 20 steps, terminal 0

    o.method = HEUN_SAMPLE_METHOD;
    Plan p = sd::step_budget::plan(o, probe, nullptr);
    CHECK_EQ(p.error, sd::step_budget::None);
    CHECK_EQ(p.steps.size(), 20);
    CHECK_EQ(p.total_evaluations, 39);

    o.method = DPM2_SAMPLE_METHOD;
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).total_evaluations, 39);

    o.method = DPMPP2S_A_SAMPLE_METHOD;
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).total_evaluations, 39);

    o.method = EULER_SAMPLE_METHOD;
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).total_evaluations, 20);

    o.method = DPMPP2M_SAMPLE_METHOD;
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).total_evaluations, 20);

    // --sigmas style schedule
    o.method = HEUN_SAMPLE_METHOD;
    o.sigmas = {14.61f, 7.8f, 3.5f, 0.0f};
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).total_evaluations, 5);

    // Equal adjacent sigmas skip the Euler evaluation entirely.
    o.method = EULER_SAMPLE_METHOD;
    o.sigmas = {3.0f, 2.0f, 2.0f, 1.0f, 0.0f};
    p = sd::step_budget::plan(o, probe, nullptr);
    CHECK_EQ(p.error, sd::step_budget::None);
    CHECK_EQ(p.steps.size(), 4);
    CHECK_EQ(p.steps[1].eval_count, 0);
    CHECK_EQ(p.steps[1].first_eval_step_code, 0);
    CHECK_EQ(p.steps[1].completion_step_code, 0);
    CHECK_EQ(p.total_evaluations, 3);
}

static void test_step_codes() {
    MockProbe probe;
    probe.available = 1ull << 40;

    Options o = base_options();
    o.method  = HEUN_SAMPLE_METHOD;
    o.sigmas  = {2.0f, 1.0f, 0.5f, 0.0f};
    Plan p    = sd::step_budget::plan(o, probe, nullptr);
    CHECK_EQ(p.error, sd::step_budget::None);
    CHECK_EQ(p.steps.size(), 3);
    // Two-evaluation steps: negative first-eval code, positive completion code.
    CHECK_EQ(p.steps[0].eval_count, 2);
    CHECK_EQ(p.steps[0].first_eval_step_code, -1);
    CHECK_EQ(p.steps[0].completion_step_code, 1);
    CHECK_EQ(p.steps[1].eval_count, 2);
    CHECK_EQ(p.steps[1].first_eval_step_code, -2);
    CHECK_EQ(p.steps[1].completion_step_code, 2);
    // Terminal step with sigma_next == 0 evaluates once.
    CHECK_EQ(p.steps[2].eval_count, 1);
    CHECK_EQ(p.steps[2].first_eval_step_code, 3);
    CHECK_EQ(p.steps[2].completion_step_code, 3);
}

static void test_dpmpp2s_flow_first_step() {
    MockProbe probe;
    probe.available = 1ull << 40;

    Options o       = base_options();
    o.method        = DPMPP2S_A_SAMPLE_METHOD;
    o.flow_denoiser = true;
    o.sigmas        = {1.0f, 0.5f, 0.0f};

    Plan p = sd::step_budget::plan(o, probe, nullptr);
    CHECK_EQ(p.error, sd::step_budget::None);
    CHECK_EQ(p.steps.size(), 2);
    CHECK_EQ(p.steps[0].eval_count, 1);
    CHECK_EQ(p.steps[0].first_eval_step_code, 1);
    CHECK_EQ(p.steps[0].completion_step_code, 1);
    CHECK_EQ(p.total_evaluations, 2);

    // Without the flow denoiser the first step needs both evaluations.
    o.flow_denoiser = false;
    p               = sd::step_budget::plan(o, probe, nullptr);
    CHECK_EQ(p.steps[0].eval_count, 2);
    CHECK_EQ(p.steps[0].first_eval_step_code, -1);
    CHECK_EQ(p.total_evaluations, 3);

    // Flow denoiser but sigma not at the 1.0 special point: two evaluations.
    o.flow_denoiser = true;
    o.sigmas        = {0.9f, 0.5f, 0.0f};
    p               = sd::step_budget::plan(o, probe, nullptr);
    CHECK_EQ(p.steps[0].eval_count, 2);
}

static void test_preview() {
    MockProbe probe;
    probe.available = 1ull << 40;

    Options o = base_options();
    o.method  = EULER_SAMPLE_METHOD;
    o.sigmas  = {4.0f, 3.0f, 2.0f, 1.0f, 0.0f};  // 4 steps

    o.preview_interval = 1;
    Plan p             = sd::step_budget::plan(o, probe, nullptr);
    for (const Step& step : p.steps) {
        CHECK_TRUE(step.preview);
    }

    o.preview_interval = -2;
    p                  = sd::step_budget::plan(o, probe, nullptr);
    CHECK_TRUE(!p.steps[0].preview);
    CHECK_TRUE(p.steps[1].preview);
    CHECK_TRUE(!p.steps[2].preview);
    CHECK_TRUE(!p.steps[3].preview);

    o.preview_interval    = 0;
    o.preview_final_step  = true;
    p                     = sd::step_budget::plan(o, probe, nullptr);
    CHECK_TRUE(!p.steps[0].preview);
    CHECK_TRUE(!p.steps[1].preview);
    CHECK_TRUE(!p.steps[2].preview);
    CHECK_TRUE(p.steps[3].preview);

    o.preview_final_step = false;
    p                    = sd::step_budget::plan(o, probe, nullptr);
    for (const Step& step : p.steps) {
        CHECK_TRUE(!step.preview);
    }

    // Zero-evaluation steps never preview.
    o.preview_interval = 1;
    o.sigmas           = {3.0f, 2.0f, 2.0f, 1.0f, 0.0f};
    p                  = sd::step_budget::plan(o, probe, nullptr);
    CHECK_TRUE(p.steps[0].preview);
    CHECK_TRUE(!p.steps[1].preview);
    CHECK_TRUE(p.steps[2].preview);
    CHECK_TRUE(p.steps[3].preview);

    // Preview decisions use the completion code, not the negative first-eval
    // code: with interval -1 the first Heun step still previews.
    o.method           = HEUN_SAMPLE_METHOD;
    o.preview_interval = -1;
    o.sigmas           = {2.0f, 1.0f, 0.0f};
    p                  = sd::step_budget::plan(o, probe, nullptr);
    CHECK_EQ(p.steps[0].first_eval_step_code, -1);
    CHECK_TRUE(p.steps[0].preview);
    CHECK_TRUE(!p.steps[1].preview);
}

static void test_validation() {
    MockProbe probe;
    probe.available = 1ull << 40;

    Options o = base_options();

    // Step count
    o.sigmas = {1.0f};
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).error, sd::step_budget::InvalidStepCount);
    o.sigmas = {};
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).error, sd::step_budget::InvalidStepCount);

    // GITS needs at least two steps.
    o.scheduler = GITS_SCHEDULER;
    o.sigmas    = {1.0f, 0.0f};
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).error, sd::step_budget::InvalidStepCount);
    o.sigmas = {2.0f, 1.0f, 0.0f};
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).error, sd::step_budget::None);
    o.scheduler = DISCRETE_SCHEDULER;

    // Method bounds are checked before sigma contents.
    o.sigmas = {1.0f};
    o.method = static_cast<sample_method_t>(-1);
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).error, sd::step_budget::InvalidStepCount);
    o.sigmas = {std::nanf(""), 0.0f};
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).error, sd::step_budget::UnsupportedMethod);
    o.method = static_cast<sample_method_t>(SAMPLE_METHOD_COUNT);
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).error, sd::step_budget::UnsupportedMethod);
    o.method = EULER_SAMPLE_METHOD;

    // Non-finite sigmas
    o.sigmas = {std::nanf(""), 0.0f};
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).error, sd::step_budget::NonFiniteSigma);
    o.sigmas = {2.0f, INFINITY, 0.0f};
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).error, sd::step_budget::NonFiniteSigma);

    // Non-positive head sigma
    o.sigmas = {0.0f, 0.0f};
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).error, sd::step_budget::NegativeSigma);
    o.sigmas = {-1.0f, -2.0f};
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).error, sd::step_budget::NegativeSigma);

    // Descending order
    o.sigmas = {2.0f, 3.0f, 1.0f, 0.0f};
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).error, sd::step_budget::ScheduleNotDescending);

    // Terminal sigma
    o.sigmas = {2.0f, 1.0f, 0.5f};
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).error, sd::step_budget::TerminalSigmaNotZero);
    o.terminal_sigma_is_zero = false;
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).error, sd::step_budget::None);
    o.terminal_sigma_is_zero = true;

    // Budget must exceed both reserves.
    o.sigmas       = {2.0f, 1.0f, 0.0f};
    o.budget_bytes = 150;  // == scratch + headroom
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).error, sd::step_budget::InvalidBudget);
    o.budget_bytes = 149;
    CHECK_EQ(sd::step_budget::plan(o, probe, nullptr).error, sd::step_budget::InvalidBudget);
    o.budget_bytes = 151;
    CHECK_TRUE(sd::step_budget::plan(o, probe, nullptr).error != sd::step_budget::InvalidBudget);

    // Failed validations produce no steps.
    o.budget_bytes = 0;
    o.sigmas       = {1.0f};
    Plan p         = sd::step_budget::plan(o, probe, nullptr);
    CHECK_TRUE(p.steps.empty());
}

static void test_admission_evaluate_exact_budget() {
    MockProbe probe;
    probe.available = 1ull << 40;

    Options o      = base_options();
    o.sigmas       = {2.0f, 1.0f, 0.0f};
    o.budget_bytes = 150 + 1500;  // reserves + required, exactly

    Plan p = sd::step_budget::plan(o, probe, nullptr);
    CHECK_EQ(p.error, sd::step_budget::None);
    CHECK_EQ(p.steps.size(), 2);
    CHECK_EQ(p.steps[0].admission, sd::step_budget::Evaluate);
    CHECK_EQ(p.steps[0].usable_bytes, 1500);
    CHECK_EQ(p.steps[0].required_bytes, 1500);
    CHECK_EQ(p.steps[0].reclaimed_bytes, 0);
    CHECK_EQ(probe.evict_calls, 0);
    CHECK_EQ(p.peak_required_bytes, 1500);
}

static void test_reclaim_order() {
    Options o      = base_options();
    o.sigmas       = {2.0f, 0.0f};  // single step: evicted sequence stays exact
    o.budget_bytes = 150 + 1250;  // usable 1250, required 1500 -> reclaim 250

    std::vector<WeightBlock> blocks = {
        make_block("prot", 1000, 0, 0, false, true),   // protected: never a candidate
        make_block("dem", 100, 0, 0, true),            // demotion sorts last
        make_block("e", 200, 3, 2),
        make_block("c", 100, 3, 2),
        make_block("d", 200, 3, 2),
        make_block("b", 100, 3, 1),
        make_block("a", 100, 0, 5),
    };

    MockProbe probe;
    probe.available = 1ull << 40;
    probe.blocks    = blocks;

    Plan p = sd::step_budget::plan(o, probe, nullptr);
    CHECK_EQ(p.error, sd::step_budget::None);
    CHECK_EQ(p.steps[0].admission, sd::step_budget::ReclaimThenEvaluate);
    // Order: next_use 0 first ("a"), then epoch 1 ("b"), then 200-byte blocks
    // by name ("d" before "e"); 100+100+200 covers the missing 250.
    CHECK_EQ(probe.evicted.size(), 3);
    CHECK_STR(probe.evicted[0], "a");
    CHECK_STR(probe.evicted[1], "b");
    CHECK_STR(probe.evicted[2], "d");
    CHECK_EQ(p.steps[0].reclaimed_bytes, 400);

    // Same input in a scrambled container order must give the same sequence.
    MockProbe probe2;
    probe2.available = 1ull << 40;
    probe2.blocks    = {blocks[3], blocks[6], blocks[1], blocks[5], blocks[0], blocks[4], blocks[2]};
    Plan p2          = sd::step_budget::plan(o, probe2, nullptr);
    CHECK_EQ(p2.steps[0].admission, sd::step_budget::ReclaimThenEvaluate);
    CHECK_EQ(probe2.evicted.size(), 3);
    CHECK_STR(probe2.evicted[0], "a");
    CHECK_STR(probe2.evicted[1], "b");
    CHECK_STR(probe2.evicted[2], "d");
}

static void test_protected_block_never_evicted() {
    MockProbe probe;
    probe.available = 1ull << 40;
    probe.blocks    = {make_block("prot", 100000, 0, 0, false, true)};

    Options o      = base_options();
    o.sigmas       = {2.0f, 1.0f, 0.0f};
    o.budget_bytes = 150 + 1250;

    Plan p = sd::step_budget::plan(o, probe, nullptr);
    CHECK_EQ(p.error, sd::step_budget::BudgetExhausted);
    CHECK_EQ(p.steps[0].admission, sd::step_budget::Rejected);
    CHECK_EQ(probe.evict_calls, 0);
    CHECK_EQ(p.rejected_step, 1);
}

static void test_rejected_when_only_demotion_blocks() {
    MockProbe probe;
    probe.available = 1ull << 40;
    probe.blocks    = {make_block("dem", 100, 0, 0, true)};

    Options o      = base_options();
    o.sigmas       = {2.0f, 1.0f, 0.0f};
    o.budget_bytes = 150 + 1250;  // needs 250, only 100 reclaimable

    Plan p = sd::step_budget::plan(o, probe, nullptr);
    CHECK_EQ(p.error, sd::step_budget::BudgetExhausted);
    CHECK_EQ(p.steps.size(), 1);  // no further steps are planned
    CHECK_EQ(p.steps[0].admission, sd::step_budget::Rejected);
    CHECK_EQ(p.steps[0].error, sd::step_budget::BudgetExhausted);
    CHECK_EQ(p.steps[0].reclaimed_bytes, 100);  // completed evictions are kept
    CHECK_EQ(p.rejected_step, 1);
    CHECK_EQ(probe.evicted.size(), 1);
    CHECK_STR(probe.evicted[0], "dem");
}

static void test_segmented() {
    MockProbe probe;
    probe.available = 1ull << 40;

    Options o                = base_options();
    o.sigmas                 = {2.0f, 1.0f, 0.0f};
    o.segment_compute_bytes  = 400;  // segment need = 400 + 500 = 900
    o.budget_bytes           = 150 + 1000;  // usable 1000 < required 1500

    Plan p = sd::step_budget::plan(o, probe, nullptr);
    CHECK_EQ(p.error, sd::step_budget::None);
    CHECK_EQ(p.steps[0].admission, sd::step_budget::Segmented);
    CHECK_EQ(p.steps[0].required_bytes, 1500);  // full-graph value kept
    CHECK_EQ(p.steps[0].usable_bytes, 1000);
    CHECK_EQ(p.steps[0].reclaimed_bytes, 0);
    CHECK_EQ(probe.evict_calls, 0);

    // Segmentation that itself needs reclamation.
    MockProbe probe2;
    probe2.available = 1ull << 40;
    probe2.blocks    = {make_block("x", 200, 1)};
    o.sigmas         = {2.0f, 0.0f};
    o.budget_bytes   = 150 + 800;  // usable 800 < segment need 900
    p                = sd::step_budget::plan(o, probe2, nullptr);
    CHECK_EQ(p.steps[0].admission, sd::step_budget::Segmented);
    CHECK_EQ(p.steps[0].reclaimed_bytes, 200);
    CHECK_EQ(probe2.evicted.size(), 1);
}

static void test_device_report_wrapped() {
    MockProbe probe;
    probe.available = 1ull << 40;
    probe.wrapped   = true;

    Options o = base_options();
    o.sigmas  = {2.0f, 1.0f, 0.0f};

    // No explicit budget: refuse to plan.
    Plan p = sd::step_budget::plan(o, probe, nullptr);
    CHECK_EQ(p.error, sd::step_budget::CapacityUnavailable);
    CHECK_TRUE(p.steps.empty());

    // Explicit budget: ignore the device reading (0 free) and plan on budget.
    probe.available = 0;
    o.budget_bytes  = 150 + 1500;
    p               = sd::step_budget::plan(o, probe, nullptr);
    CHECK_EQ(p.error, sd::step_budget::None);
    CHECK_EQ(p.steps.size(), 2);
    CHECK_EQ(p.steps[0].admission, sd::step_budget::Evaluate);
    CHECK_EQ(p.steps[0].usable_bytes, 1500);
}

static void test_saturation() {
    MockProbe probe;
    probe.available = 1ull << 40;

    Options o        = base_options();
    o.sigmas         = {2.0f, 1.0f, 0.0f};
    o.compute_bytes  = SIZE_MAX - 8;
    o.weight_bytes   = 16;

    Plan p = sd::step_budget::plan(o, probe, nullptr);
    CHECK_EQ(p.steps[0].required_bytes, SIZE_MAX);

    // Reserve subtraction wraps down to 0, never to a huge value.
    MockProbe probe2;
    probe2.available = 10;
    Options o2       = base_options();
    o2.sigmas        = {2.0f, 1.0f, 0.0f};
    p                = sd::step_budget::plan(o2, probe2, nullptr);
    CHECK_EQ(p.steps[0].usable_bytes, 0);
    CHECK_EQ(p.steps[0].admission, sd::step_budget::Rejected);
}

static void test_evict_failure_aborts() {
    MockProbe probe;
    probe.available = 1650;  // step 1 evaluates; the hook cuts it off later
    probe.blocks    = {make_block("x", 1000, 1)};
    probe.on_available = [](MockProbe& self, size_t call) {
        if (call == 1) {
            self.available    = 0;
            self.evict_result = false;
        }
    };

    Options o = base_options();
    o.sigmas  = {3.0f, 2.0f, 1.0f, 0.0f};

    Plan p = sd::step_budget::plan(o, probe, nullptr);
    CHECK_EQ(p.error, sd::step_budget::CapacityUnavailable);
    CHECK_EQ(p.steps.size(), 1);  // recorded steps are kept
    CHECK_EQ(p.steps[0].admission, sd::step_budget::Evaluate);
}

static void test_ledger_basics() {
    DeviceLedger ledger(1000);
    CHECK_EQ(ledger.budget_bytes(), 1000);
    CHECK_EQ(ledger.committed_bytes(), 0);
    CHECK_EQ(ledger.remaining_bytes(), 1000);
    CHECK_TRUE(ledger.fits(1000));
    CHECK_TRUE(!ledger.fits(1001));

    CHECK_EQ(ledger.reserve("a", 600), sd::step_budget::None);
    CHECK_EQ(ledger.committed_bytes(), 600);
    CHECK_EQ(ledger.remaining_bytes(), 400);
    CHECK_EQ(ledger.peak_bytes(), 600);

    // Over budget: error, state unchanged.
    CHECK_EQ(ledger.reserve("b", 500), sd::step_budget::BudgetExhausted);
    CHECK_EQ(ledger.committed_bytes(), 600);

    // Overflow: error, state unchanged.
    CHECK_EQ(ledger.reserve("c", SIZE_MAX), sd::step_budget::InvalidSize);
    CHECK_EQ(ledger.committed_bytes(), 600);

    // Underflow: error, state unchanged.
    CHECK_EQ(ledger.release("a", 601), sd::step_budget::LedgerUnderflow);
    CHECK_EQ(ledger.release("missing", 1), sd::step_budget::LedgerUnderflow);
    CHECK_EQ(ledger.committed_bytes(), 600);

    CHECK_EQ(ledger.release("a", 600), sd::step_budget::None);
    CHECK_EQ(ledger.committed_bytes(), 0);
    CHECK_EQ(ledger.peak_bytes(), 600);  // peak is historical

    CHECK_EQ(ledger.reserve("a", 300), sd::step_budget::None);
    CHECK_EQ(ledger.peak_bytes(), 600);
    CHECK_EQ(ledger.release("a", 300), sd::step_budget::None);

    // Unbounded ledger
    DeviceLedger unbounded(0);
    CHECK_EQ(unbounded.remaining_bytes(), SIZE_MAX);
    CHECK_TRUE(unbounded.fits(SIZE_MAX));
    CHECK_EQ(unbounded.reserve("x", 1ull << 40), sd::step_budget::None);
    CHECK_EQ(unbounded.committed_bytes(), 1ull << 40);
}

static void test_ledger_threads() {
    DeviceLedger ledger(4096);
    std::atomic<bool> ok{true};

    auto worker = [&ledger, &ok](int tid) {
        const std::string owner = "owner-" + std::to_string(tid);
        size_t outstanding      = 0;
        uint64_t state          = 0x9e3779b97f4a7c15ull * static_cast<uint64_t>(tid + 1);
        for (int i = 0; i < 2000; i++) {
            state       = state * 6364136223846793005ull + 1442695040888963407ull;
            size_t bytes = static_cast<size_t>(state >> 33) % 64 + 1;
            if ((i & 1) == 0 || outstanding == 0) {
                Error e = ledger.reserve(owner, bytes);
                if (e == sd::step_budget::None) {
                    outstanding += bytes;
                } else if (e != sd::step_budget::BudgetExhausted &&
                           e != sd::step_budget::InvalidSize) {
                    ok = false;
                }
            } else {
                size_t amount = std::min(bytes, outstanding);
                if (ledger.release(owner, amount) != sd::step_budget::None) {
                    ok = false;
                } else {
                    outstanding -= amount;
                }
            }
        }
        if (outstanding > 0 && ledger.release(owner, outstanding) != sd::step_budget::None) {
            ok = false;
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < 8; t++) {
        threads.emplace_back(worker, t);
    }
    for (std::thread& thread : threads) {
        thread.join();
    }

    CHECK_TRUE(ok.load());
    CHECK_EQ(ledger.committed_bytes(), 0);
    CHECK_TRUE(ledger.peak_bytes() <= 4096);
}

static void test_ledger_in_plan() {
    MockProbe probe;
    probe.available = 1ull << 40;
    probe.blocks    = {make_block("a", 100, 1), make_block("b", 100, 2)};

    DeviceLedger ledger(1650);
    CHECK_EQ(ledger.reserve("runtime", 150), sd::step_budget::None);

    Options o      = base_options();
    o.sigmas       = {2.0f, 0.0f};
    o.budget_bytes = 1650;

    Plan p = sd::step_budget::plan(o, probe, &ledger);
    CHECK_EQ(p.error, sd::step_budget::None);
    // remaining = 1650 - 150 committed = 1500; usable = 1500 - 150 reserves.
    CHECK_EQ(p.steps[0].usable_bytes, 1350);
    CHECK_EQ(p.steps[0].admission, sd::step_budget::ReclaimThenEvaluate);
    CHECK_EQ(p.steps[0].reclaimed_bytes, 200);
    // plan only reads the ledger.
    CHECK_EQ(ledger.committed_bytes(), 150);
}

static void test_cancel() {
    std::atomic<bool> cancel{false};

    MockProbe probe;
    probe.available    = 1ull << 40;
    probe.on_available = [&cancel](MockProbe&, size_t call) {
        if (call == 2) {
            cancel.store(true);
        }
    };

    Options o = base_options();
    o.sigmas  = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f};  // 5 Euler steps
    o.cancel  = &cancel;

    Plan p = sd::step_budget::plan(o, probe, nullptr);
    CHECK_EQ(p.error, sd::step_budget::Canceled);
    CHECK_EQ(p.steps.size(), 2);
    CHECK_EQ(p.total_evaluations, 2);
}

static void check_same_plan(const Plan& x, const Plan& y) {
    CHECK_EQ(x.error, y.error);
    CHECK_EQ(x.total_evaluations, y.total_evaluations);
    CHECK_EQ(x.peak_required_bytes, y.peak_required_bytes);
    CHECK_EQ(x.rejected_step, y.rejected_step);
    CHECK_EQ(x.steps.size(), y.steps.size());
    const size_t n = std::min(x.steps.size(), y.steps.size());
    for (size_t i = 0; i < n; i++) {
        const Step& a = x.steps[i];
        const Step& b = y.steps[i];
        CHECK_EQ(a.index, b.index);
        CHECK_TRUE(a.sigma == b.sigma);
        CHECK_TRUE(a.sigma_next == b.sigma_next);
        CHECK_EQ(a.eval_count, b.eval_count);
        CHECK_EQ(a.first_eval_step_code, b.first_eval_step_code);
        CHECK_EQ(a.completion_step_code, b.completion_step_code);
        CHECK_EQ(a.preview, b.preview);
        CHECK_EQ(a.required_bytes, b.required_bytes);
        CHECK_EQ(a.usable_bytes, b.usable_bytes);
        CHECK_EQ(a.reclaimed_bytes, b.reclaimed_bytes);
        CHECK_EQ(a.admission, b.admission);
        CHECK_EQ(a.error, b.error);
    }
}

static void test_dry_run() {
    Options o      = base_options();
    o.sigmas       = {3.0f, 2.0f, 1.0f, 0.0f};
    o.budget_bytes = 150 + 1250;  // every step reclaims

    std::vector<WeightBlock> blocks = {
        make_block("a", 100, 1),
        make_block("b", 100, 2),
        make_block("c", 300, 3),
    };

    MockProbe real_probe;
    real_probe.available = 1ull << 40;
    real_probe.blocks    = blocks;
    Plan real            = sd::step_budget::plan(o, real_probe, nullptr);
    CHECK_TRUE(real_probe.evict_calls > 0);

    o.dry_run = true;
    MockProbe dry_probe;
    dry_probe.available = 1ull << 40;
    dry_probe.blocks    = blocks;
    Plan dry            = sd::step_budget::plan(o, dry_probe, nullptr);

    CHECK_EQ(dry_probe.evict_calls, 0);
    check_same_plan(real, dry);
}

static std::string capture_stderr(const std::function<void()>& fn) {
    std::fflush(stderr);
    FILE* tmp = std::tmpfile();
    if (tmp == nullptr) {
        fn();
        return "";
    }
    int old_fd = sd_dup(sd_fileno(stderr));
    sd_dup2(sd_fileno(tmp), sd_fileno(stderr));
    fn();
    std::fflush(stderr);
    sd_dup2(old_fd, sd_fileno(stderr));
    sd_close(old_fd);
    std::rewind(tmp);
    std::string out;
    char buf[512];
    while (std::fgets(buf, sizeof(buf), tmp) != nullptr) {
        out += buf;
    }
    std::fclose(tmp);
    return out;
}

static void test_log_prefix() {
    Options o      = base_options();
    o.sigmas       = {2.0f, 1.0f, 0.0f};
    o.budget_bytes = 150 + 1250;

    MockProbe probe;
    probe.available = 1ull << 40;
    probe.blocks    = {make_block("dem", 100, 0, 0, true)};

    std::string out = capture_stderr([&] { sd::step_budget::plan(o, probe, nullptr); });
    CHECK_TRUE(out.find("step-budget: step=1 eval=1 required=1500 usable=1250 "
                        "reclaimed=100 admission=3\n") != std::string::npos);

    // Segmented steps log the same prefix.
    MockProbe probe2;
    probe2.available = 1ull << 40;
    o.segment_compute_bytes = 400;
    o.budget_bytes          = 150 + 1000;
    out = capture_stderr([&] { sd::step_budget::plan(o, probe2, nullptr); });
    CHECK_TRUE(out.find("step-budget: step=1 eval=1 required=1500 usable=1000 "
                        "reclaimed=0 admission=2\n") != std::string::npos);
}

static void test_plan_is_deterministic() {
    Options o      = base_options();
    o.sigmas       = countdown_sigmas(20, 21);
    o.budget_bytes = 150 + 1250;

    MockProbe probe1;
    probe1.available = 1ull << 40;
    probe1.blocks    = {make_block("a", 100, 1), make_block("b", 200, 0, 3),
                        make_block("c", 200, 0, 2)};
    Plan p1 = sd::step_budget::plan(o, probe1, nullptr);

    MockProbe probe2;
    probe2.available = 1ull << 40;
    probe2.blocks    = probe1.blocks;
    Plan p2          = sd::step_budget::plan(o, probe2, nullptr);

    check_same_plan(p1, p2);
}

static void test_perf_scale() {
    // 10000 steps with 200 evictable blocks resident: the first steps reclaim
    // (one fetch, one sort), the freed bytes then cover the remaining steps.
    MockProbe probe;
    probe.available          = 1400;  // usable 1250 < required 1500
    probe.evict_frees_memory = true;
    probe.blocks.reserve(200);
    for (int i = 0; i < 200; i++) {
        probe.blocks.push_back(
            make_block("blk-" + std::to_string(i), 100, static_cast<size_t>(i % 7),
                       static_cast<uint64_t>(i)));
    }

    Options o = base_options();
    o.method  = EULER_SAMPLE_METHOD;
    o.sigmas  = countdown_sigmas(10000, 10001);

    const auto start = std::chrono::steady_clock::now();
    Plan p           = sd::step_budget::plan(o, probe, nullptr);
    const auto end   = std::chrono::steady_clock::now();
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    CHECK_EQ(p.error, sd::step_budget::None);
    CHECK_EQ(p.steps.size(), 10000);
    CHECK_EQ(p.total_evaluations, 10000);
    CHECK_EQ(p.steps[0].admission, sd::step_budget::ReclaimThenEvaluate);
    CHECK_EQ(p.steps.back().admission, sd::step_budget::Evaluate);
    // Generous wall-clock bound; the design target is 20 ms for this input.
    CHECK_TRUE(ms < 5000);
}

int main() {
    test_to_string();
    test_eval_counts();
    test_step_codes();
    test_dpmpp2s_flow_first_step();
    test_preview();
    test_validation();
    test_admission_evaluate_exact_budget();
    test_reclaim_order();
    test_protected_block_never_evicted();
    test_rejected_when_only_demotion_blocks();
    test_segmented();
    test_device_report_wrapped();
    test_saturation();
    test_evict_failure_aborts();
    test_ledger_basics();
    test_ledger_threads();
    test_ledger_in_plan();
    test_cancel();
    test_dry_run();
    test_log_prefix();
    test_plan_is_deterministic();
    test_perf_scale();

    std::fprintf(stderr, "%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
