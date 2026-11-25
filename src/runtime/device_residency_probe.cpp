#include "runtime/device_residency_probe.h"

namespace sd::step_budget {

    namespace {

        WeightBlock map_block(const WeightResidencyBlock& src, size_t index) {
            WeightBlock block;
            if (!src.tensors.empty() && src.tensors.front() != nullptr) {
                block.name = src.tensors.front()->name;
            } else {
                block.name = "block-" + std::to_string(index);
            }
            block.bytes             = src.bytes;
            block.last_use_epoch    = src.last_use_epoch;
            block.next_use          = src.next_use;
            block.requires_demotion = src.requires_demotion;
            block.protected_block   = false;
            return block;
        }

        bool same_block(const WeightBlock& lhs, const WeightBlock& rhs) {
            return lhs.name == rhs.name && lhs.bytes == rhs.bytes &&
                   lhs.last_use_epoch == rhs.last_use_epoch && lhs.next_use == rhs.next_use &&
                   lhs.requires_demotion == rhs.requires_demotion &&
                   lhs.protected_block == rhs.protected_block;
        }

    }  // namespace

    DeviceResidencyProbe::DeviceResidencyProbe(DeviceResidencyManager& manager,
                                               ggml_backend_t compute_backend)
        : manager_(manager), compute_backend_(compute_backend) {}

    size_t DeviceResidencyProbe::available_device_bytes() {
        if (compute_backend_ == nullptr) {
            return 0;
        }
        ggml_backend_dev_t device = ggml_backend_get_device(compute_backend_);
        if (device == nullptr) {
            return 0;
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(device, &props);
        return props.memory_free;
    }

    size_t DeviceResidencyProbe::resident_weight_bytes() {
        size_t total = 0;
        for (const WeightResidencyBlock& block :
             manager_.evictable_compute_backend_blocks(compute_backend_, {})) {
            total = block.bytes > SIZE_MAX - total ? SIZE_MAX : total + block.bytes;
        }
        return total;
    }

    std::vector<WeightBlock> DeviceResidencyProbe::evictable_blocks() {
        last_blocks_ = manager_.evictable_compute_backend_blocks(compute_backend_, {});
        std::vector<WeightBlock> blocks;
        blocks.reserve(last_blocks_.size());
        for (size_t i = 0; i < last_blocks_.size(); i++) {
            blocks.push_back(map_block(last_blocks_[i], i));
        }
        return blocks;
    }

    bool DeviceResidencyProbe::evict(const WeightBlock& block) {
        for (size_t i = 0; i < last_blocks_.size(); i++) {
            if (!same_block(map_block(last_blocks_[i], i), block)) {
                continue;
            }
            WeightResidencyBlock target = last_blocks_[i];
            last_blocks_.erase(last_blocks_.begin() + i);
            return manager_.evict_compute_backend_block(target);
        }
        return false;
    }

    bool DeviceResidencyProbe::device_report_wrapped() {
        if (compute_backend_ == nullptr) {
            return false;
        }
        ggml_backend_dev_t device = ggml_backend_get_device(compute_backend_);
        if (device == nullptr) {
            return false;
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(device, &props);
        return props.memory_free > props.memory_total;
    }

}  // namespace sd::step_budget
