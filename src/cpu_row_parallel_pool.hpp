#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace tao::dual {
// N participating threads: the caller plus N-1 persistent workers.
//
// Rows are split into N contiguous chunks [rows*k/N, rows*(k+1)/N). Every row is
// still produced by exactly the same per-row kernel call, so the result is
// bitwise identical to the serial path for every N. No dot product is split and
// no per-row accumulation order changes. **Synchronisation mechanism does not
// affect numerics**: the row partition depends only on (rows, N).
//
// Chunk 0 runs on the caller; chunks 1..N-1 run on workers. Tasks are
// synchronous: all captures stay alive until run() returns, including on
// exception paths. No concurrent destruction, recursive run(), or concurrent
// run() from two callers.
//
// Two sync mechanisms are available, selected once at construction:
//   * sleep (default, historical): condition_variable handoff. Each dispatch
//     parks/wakes N-1 threads through the futex path.
//   * spin (TAO_POOL_SPIN=1): atomic generation counters with _mm_pause().
//     Measured motivation: at ~32 dispatches per token the sleep path costs
//     ~15 us per barrier, which was ~480 us of a 610 us token budget and was
//     the reason thread scaling saturated at 1.10x over 16 threads.
class CpuRowParallelPool {
public:
    static constexpr size_t minimum_elements = 262144;
    // Upper bound used for fixed-size stack scratch in the pool and in the
    // greedy head. Kept small so those frames stay off the stack-probe path.
    static constexpr unsigned max_threads = 16;

private:
    struct Slot {
        size_t begin = 0, end = 0;
        std::exception_ptr error;
    };
    static void pause() {
#if defined(_MSC_VER) || defined(__GNUC__)
        _mm_pause();
#endif
    }
    std::mutex callers_, mutex_;
    std::condition_variable ready_, done_;
    bool stopping_ = false;
    unsigned generation_ = 0;
    size_t remaining_ = 0;
    void* context_ = nullptr;
    void (*invoke_)(void*, size_t, size_t, size_t) = nullptr;
    std::vector<Slot> slots_;
    std::vector<std::thread> workers_;
    size_t min_elements_ = minimum_elements;
    bool spin_ = false;
    std::atomic<unsigned> spin_gen_{0};
    std::atomic<unsigned> spin_done_{0};
    std::atomic<bool> spin_stop_{false};

    void loop(unsigned index) {
        if (spin_) {
            unsigned seen = 0;
            for (;;) {
                unsigned g;
                for (;;) {
                    if (spin_stop_.load(std::memory_order_relaxed)) return;
                    g = spin_gen_.load(std::memory_order_acquire);
                    if (g != seen) break;
                    pause();
                }
                // Defensive re-check. A worker preempted between the flag check
                // above and the generation load would otherwise break out and
                // run a task against slots that are about to be cleared. That
                // race caused 0xC0000005 at 8 threads in 3 of 5 runs
                // (02-cpu-decode-throughput.md section 24).
                if (spin_stop_.load(std::memory_order_acquire)) return;
                seen = g;
                const size_t begin = slots_[index].begin, end = slots_[index].end;
                auto fn = invoke_;
                auto ctx = context_;
                std::exception_ptr error;
                try { fn(ctx, index + 1u, begin, end); } catch (...) { error = std::current_exception(); }
                slots_[index].error = error;
                spin_done_.fetch_add(1u, std::memory_order_acq_rel);
            }
        }
        std::unique_lock<std::mutex> lock(mutex_);
        unsigned seen = 0;
        for (;;) {
            ready_.wait(lock, [&] { return stopping_ || generation_ != seen; });
            if (stopping_) return;
            seen = generation_;
            const size_t begin = slots_[index].begin, end = slots_[index].end;
            auto fn = invoke_;
            auto ctx = context_;
            lock.unlock();
            std::exception_ptr error;
            try { fn(ctx, index + 1u, begin, end); } catch (...) { error = std::current_exception(); }
            lock.lock();
            slots_[index].error = error;
            if (--remaining_ == 0) done_.notify_one();
        }
    }
    void start_workers(unsigned total) {
        workers_.reserve(total ? total - 1u : 0u);
        for (unsigned i = 0; i + 1u < total; ++i) workers_.emplace_back([this, i] { loop(i); });
        slots_.assign(workers_.size(), Slot{});
    }
    void stop_workers() {
        if (spin_) {
            // Deliberately do NOT bump the generation here. The worker notices
            // the flag in its inner poll loop, so joining is sufficient. Bumping
            // it was what allowed a worker to observe a fresh generation while
            // the stop flag was already set -- and then execute a bogus task.
            spin_stop_.store(true, std::memory_order_release);
            for (auto& t : workers_) if (t.joinable()) t.join();
            workers_.clear();
            slots_.clear();
            spin_stop_.store(false, std::memory_order_relaxed);
            spin_done_.store(0u, std::memory_order_relaxed);
            // Reset the generation too: workers restart with seen=0, and a
            // non-zero generation here would make each of them fire one bogus
            // task against a stale context on the very first dispatch.
            spin_gen_.store(0u, std::memory_order_relaxed);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& t : workers_) if (t.joinable()) t.join();
        workers_.clear();
        slots_.clear();
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = false;
        generation_ = 0;
    }

public:
    explicit CpuRowParallelPool(unsigned threads = 2) {
        const char* sp = std::getenv("TAO_POOL_SPIN");
        spin_ = sp != nullptr && *sp && *sp != '0';
        if (threads < 1u) threads = 1u;
        if (threads > max_threads) threads = max_threads;
        start_workers(threads);
    }
    CpuRowParallelPool(const CpuRowParallelPool&) = delete;
    CpuRowParallelPool& operator=(const CpuRowParallelPool&) = delete;
    ~CpuRowParallelPool() { stop_workers(); }

    unsigned thread_count() const { return unsigned(workers_.size()) + 1u; }
    size_t minimum_dispatch_cost() const { return min_elements_; }
    void set_minimum_dispatch_cost(size_t v) {
        std::lock_guard<std::mutex> caller(callers_);
        min_elements_ = v;
    }
    bool spinning() const { return spin_; }

    // Reconfiguration is only legal between tasks (no dispatch in flight).
    void set_threads(unsigned threads) {
        std::lock_guard<std::mutex> caller(callers_);
        if (threads < 1u) threads = 1u;
        if (threads > max_threads) threads = max_threads;
        if (threads == thread_count()) return;
        stop_workers();
        start_workers(threads);
    }

    // fn is invoked as fn(chunk_index, begin_row, end_row). Chunk 0 always runs
    // on the caller, so a serial fallback is exactly fn(0, 0, rows).
    template <class F> void run(size_t rows, size_t cols, F fn) {
        const unsigned n = thread_count();
        // 串行快路径只调用 fn，不触碰 context_/invoke_/slots_，因此不需要 callers_。
        // 此前在入口就加锁，使「每线程一个模型实例并发解码」也被迫串行 —— 这是
        // 多核利用率上不去的直接原因之一。
        if (n <= 1u || rows < 2u || cols == 0u ||
            rows < (min_elements_ / cols + (min_elements_ % cols != 0))) {
            fn(size_t(0), size_t(0), rows);
            return;
        }
        std::lock_guard<std::mutex> caller(callers_);
        size_t bounds[max_threads + 1];
        for (unsigned k = 0; k <= n; ++k) bounds[k] = rows * size_t(k) / size_t(n);
        const unsigned workers_used = n - 1u;
        if (spin_) {
            context_ = &fn;
            invoke_ = [](void* p, size_t chunk, size_t begin, size_t end) {
                (*static_cast<F*>(p))(chunk, begin, end);
            };
            for (unsigned k = 1; k < n; ++k) {
                slots_[k - 1u].begin = bounds[k];
                slots_[k - 1u].end = bounds[k + 1u];
                slots_[k - 1u].error = nullptr;
            }
            spin_done_.store(0u, std::memory_order_relaxed);
            spin_gen_.fetch_add(1u, std::memory_order_release);
            std::exception_ptr caller_error;
            try { fn(size_t(0), bounds[0], bounds[1]); } catch (...) { caller_error = std::current_exception(); }
            while (spin_done_.load(std::memory_order_acquire) != workers_used) pause();
            if (caller_error) std::rethrow_exception(caller_error);
            for (unsigned k = 0; k < workers_used; ++k)
                if (slots_[k].error) std::rethrow_exception(slots_[k].error);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            context_ = &fn;
            invoke_ = [](void* p, size_t chunk, size_t begin, size_t end) {
                (*static_cast<F*>(p))(chunk, begin, end);
            };
            remaining_ = workers_used;
            for (unsigned k = 1; k < n; ++k) {
                slots_[k - 1u].begin = bounds[k];
                slots_[k - 1u].end = bounds[k + 1u];
                slots_[k - 1u].error = nullptr;
            }
            ++generation_;
        }
        ready_.notify_all();
        std::exception_ptr caller_error;
        try { fn(size_t(0), bounds[0], bounds[1]); } catch (...) { caller_error = std::current_exception(); }
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return remaining_ == 0; });
        std::exception_ptr worker_errors[max_threads];
        for (unsigned k = 0; k < workers_used; ++k) worker_errors[k] = slots_[k].error;
        lock.unlock();
        if (caller_error) std::rethrow_exception(caller_error);
        for (unsigned k = 0; k < workers_used; ++k) if (worker_errors[k]) std::rethrow_exception(worker_errors[k]);
    }
};
}
