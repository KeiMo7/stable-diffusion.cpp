#ifndef __SD_RUNTIME_DEVICE_RESIDENCY_PROBE_H__
#define __SD_RUNTIME_DEVICE_RESIDENCY_PROBE_H__

#include "device_residency_manager.h"
#include "runtime/step_budget.h"

namespace sd::step_budget {

    // Adapts a DeviceResidencyManager plus its compute backend to the
    // ResidencyProbe interface the step budget planner consumes.
    class DeviceResidencyProbe : public ResidencyProbe {
    public:
        DeviceResidencyProbe(DeviceResidencyManager& manager, ggml_backend_t compute_backend);

        size_t available_device_bytes() override;
        size_t resident_weight_bytes() override;
        std::vector<WeightBlock> evictable_blocks() override;
        bool evict(const WeightBlock& block) override;
        bool device_report_wrapped() override;

    private:
        DeviceResidencyManager& manager_;
        ggml_backend_t compute_backend_;
        // Snapshot of the last evictable_blocks() call, used to translate a
        // WeightBlock back to its WeightResidencyBlock on evict().
        std::vector<WeightResidencyBlock> last_blocks_;
    };

}  // namespace sd::step_budget

#endif  // __SD_RUNTIME_DEVICE_RESIDENCY_PROBE_H__
