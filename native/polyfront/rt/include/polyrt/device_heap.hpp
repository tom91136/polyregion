#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fmt/format.h"

#include "polyinvoke/runtime.h"
#include "polyrt/rt.h"

namespace polyregion::polyrt {

class DeviceHeapExhausted : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

class DeviceHeap {
private:
  invoke::Device &device;
  invoke::DeviceQueue &queue;
  std::function<void()> settle;
  size_t slack;
  invoke::HeapLimits limits;
  HeapCounters &counters;
  std::shared_ptr<std::shared_mutex> readers;

  uintptr_t base = 0;
  size_t committed = 0;
  size_t liveBlocks = 0;
  std::map<size_t, size_t> freeByOffset;
  std::set<std::pair<size_t, size_t>> freeBySize;
  std::map<size_t, size_t> used;

  [[nodiscard]] size_t blockSize(const size_t bytes) const {
    const auto padded = bytes + 1 + slack;
    if (padded < bytes) return std::numeric_limits<size_t>::max();
    const auto rounded = (padded + limits.alignment - 1) / limits.alignment * limits.alignment;
    return rounded < padded ? std::numeric_limits<size_t>::max() : rounded;
  }

  void addFree(size_t offset, size_t size) {
    if (const auto next = freeByOffset.lower_bound(offset); next != freeByOffset.end() && next->first == offset + size) {
      size += next->second;
      freeBySize.erase({next->second, next->first});
      freeByOffset.erase(next);
    }
    if (const auto next = freeByOffset.lower_bound(offset); next != freeByOffset.begin()) {
      if (const auto previous = std::prev(next); previous->first + previous->second == offset) {
        offset = previous->first;
        size += previous->second;
        freeBySize.erase({previous->second, previous->first});
        freeByOffset.erase(previous);
      }
    }
    freeByOffset.emplace(offset, size);
    freeBySize.emplace(size, offset);
  }

  void ensureReserved(const size_t firstBlock) {
    if (base != 0) return;
    const size_t initial = std::min(limits.maxBytes, std::bit_ceil(limits.alignment + firstBlock));
    base = device.reserveDevice(limits.maxBytes, initial);
    committed = initial;
    counters.committed = committed;
    if (committed > limits.alignment) addFree(limits.alignment, committed - limits.alignment);
  }

  [[noreturn]] void exhausted(const size_t bytes, const size_t block) const {
    throw DeviceHeapExhausted(
        fmt::format("device heap exhausted: requested {} bytes ({} with padding), {} bytes live in {} blocks, limit {} bytes", bytes, block,
                    counters.live.load(), liveBlocks, limits.maxBytes));
  }

  [[nodiscard]] size_t tailStart() const {
    const auto tail = freeByOffset.empty() ? freeByOffset.end() : std::prev(freeByOffset.end());
    return tail != freeByOffset.end() && tail->first + tail->second == committed ? tail->first : committed;
  }

  void growTo(const size_t needed) {
    const size_t target = std::min(limits.maxBytes, std::bit_ceil(needed));
    {
      const std::unique_lock exclusive(*readers);
      device.growDevice(base, target);
    }
    addFree(committed, target - committed);
    committed = target;
    counters.committed = committed;
    counters.growths++;
  }

  void grow(const size_t bytes, const size_t block) {
    const auto start = tailStart();
    if (block > limits.maxBytes || start > limits.maxBytes - block) exhausted(bytes, block);
    growTo(start + block);
  }

  void zeroPad(const size_t offset, const size_t bytes, const size_t block) {
    const std::vector<uint8_t> zeros(block - bytes, 0);
    queue.enqueueHostToDeviceAsync(zeros.data(), base, offset + bytes, zeros.size(), {});
    settle();
  }

public:
  DeviceHeap(invoke::Device &device, invoke::DeviceQueue &queue, std::function<void()> settle, const size_t slack,
             const invoke::HeapLimits limits, HeapCounters &counters,
             std::shared_ptr<std::shared_mutex> readers = std::make_shared<std::shared_mutex>())
      : device(device), queue(queue), settle(std::move(settle)), slack(slack), limits(limits), counters(counters),
        readers(std::move(readers)) {
    if (limits.alignment == 0 || (limits.alignment & (limits.alignment - 1)) != 0)
      throw std::invalid_argument("device heap alignment must be a power of two");
  }

  DeviceHeap(const DeviceHeap &) = delete;
  DeviceHeap &operator=(const DeviceHeap &) = delete;

  ~DeviceHeap() {
    if (base == 0) return;
    try {
      device.freeDevice(base);
    } catch (...) {
    }
  }

  uintptr_t allocate(const size_t bytes) {
    const auto block = blockSize(bytes);
    ensureReserved(block == std::numeric_limits<size_t>::max() ? 0 : block);
    auto fit = freeBySize.lower_bound({block, 0});
    if (fit == freeBySize.end()) {
      grow(bytes, block);
      fit = freeBySize.lower_bound({block, 0});
    }
    const auto [size, offset] = *fit;
    freeBySize.erase(fit);
    freeByOffset.erase(offset);
    if (size > block) addFree(offset + block, size - block);
    used.emplace(offset, block);
    liveBlocks++;
    counters.live += block;

    try {
      zeroPad(offset, bytes, block);
    } catch (...) {
      release(base + offset);
      throw;
    }
    return base + offset;
  }

  bool extend(const uintptr_t pointer, const size_t bytes) {
    const auto found = used.find(pointer - base);
    if (pointer < base || found == used.end()) throw std::logic_error(fmt::format("device heap does not own the block at {}", pointer));
    const auto [offset, block] = *found;
    const auto wanted = blockSize(bytes);
    if (wanted <= block) return true;
    const auto needed = wanted - block;
    auto next = freeByOffset.find(offset + block);
    if (const auto have = next == freeByOffset.end() ? 0 : next->second; have < needed) {
      if (offset + block + have != committed || wanted > limits.maxBytes || offset > limits.maxBytes - wanted) return false;
      growTo(offset + wanted);
      next = freeByOffset.find(offset + block);
    }
    const auto [spanOffset, spanSize] = *next;
    freeBySize.erase({spanSize, spanOffset});
    freeByOffset.erase(next);
    if (spanSize > needed) addFree(spanOffset + needed, spanSize - needed);
    found->second = wanted;
    counters.live += needed;
    zeroPad(offset, bytes, wanted);
    return true;
  }

  void release(const uintptr_t pointer) {
    const auto found = used.find(pointer - base);
    if (pointer < base || found == used.end()) throw std::logic_error(fmt::format("device heap does not own the block at {}", pointer));
    const auto [offset, block] = *found;
    used.erase(found);
    liveBlocks--;
    counters.live -= block;
    addFree(offset, block);
  }

  [[nodiscard]] bool owns(const uintptr_t pointer) const { return base != 0 && pointer >= base && pointer - base < committed; }
  [[nodiscard]] uintptr_t handle() const { return base; }
  [[nodiscard]] size_t committedBytes() const { return committed; }
  [[nodiscard]] size_t liveBytes() const { return counters.live.load(); }
  [[nodiscard]] size_t blocks() const { return liveBlocks; }
  [[nodiscard]] size_t alignment() const { return limits.alignment; }
};

} // namespace polyregion::polyrt
