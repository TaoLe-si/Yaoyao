#pragma once
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <cstddef>
#include <limits>
#include <chrono>
#include <cstdio>

namespace tao::dual {
// Exactly two participating threads: caller plus one persistent sleeping worker.
// Synchronous borrowed task: all captures remain alive until run() returns,
// including on exceptions. No concurrent destruction or recursive run() allowed.
class ProfileRowExecutor {
    std::mutex callers_, mutex_;
    std::condition_variable ready_, done_;
    bool pending_=false, stopping_=false;
    void* context_=nullptr;
    void (*invoke_)(void*,size_t,size_t)=nullptr;
    size_t begin_=0,end_=0;
    std::exception_ptr error_;
    std::thread worker_;
    void loop() {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            ready_.wait(lock,[this]{return stopping_||pending_;});
            if(stopping_)return;
            auto fn=invoke_;auto ctx=context_;auto begin=begin_,end=end_;
            lock.unlock();
            std::exception_ptr error;
            try {fn(ctx,begin,end);} catch (...) {error=std::current_exception();}
            lock.lock();
            error_=error;pending_=false;context_=nullptr;invoke_=nullptr;
            done_.notify_one();
        }
    }
public:
    double total_seconds=0,wait_seconds=0,caller_compute_seconds=0;size_t calls=0;
static constexpr size_t minimum_elements=262144;
    ProfileRowExecutor():worker_([this]{loop();}) {}
    ProfileRowExecutor(const ProfileRowExecutor&)=delete;
    ProfileRowExecutor& operator=(const ProfileRowExecutor&)=delete;
    ~ProfileRowExecutor() {
printf("PROFILE parallel_calls=%zu dispatch_seconds=%.6f caller_compute_plus_lock_seconds=%.6f wait_seconds=%.6f\n",calls,total_seconds,caller_compute_seconds,wait_seconds);
        {std::lock_guard<std::mutex> lock(mutex_);stopping_=true;}
        ready_.notify_one();
        if(worker_.joinable())worker_.join();
    }
    template<class F> void run(size_t rows,size_t cols,F fn) {
        std::lock_guard<std::mutex> caller(callers_);
        // Division form avoids overflow in rows*cols.
        if(rows<2||cols==0||rows<(minimum_elements/cols+(minimum_elements%cols!=0))) {
            fn(0,rows);return;
        }
        auto started=std::chrono::steady_clock::now();++calls;const size_t split=rows/2;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            context_=&fn;
            invoke_=[](void* p,size_t begin,size_t end){(*static_cast<F*>(p))(begin,end);};
            begin_=split;end_=rows;error_=nullptr;pending_=true;
        }
        ready_.notify_one();
        std::exception_ptr caller_error;
        auto compute_start=std::chrono::steady_clock::now();try {fn(0,split);} catch (...) {caller_error=std::current_exception();}
        std::unique_lock<std::mutex> lock(mutex_);
        auto wait_start=std::chrono::steady_clock::now();caller_compute_seconds+=std::chrono::duration<double>(wait_start-compute_start).count();done_.wait(lock,[this]{return !pending_;});auto finished=std::chrono::steady_clock::now();wait_seconds+=std::chrono::duration<double>(finished-wait_start).count();total_seconds+=std::chrono::duration<double>(finished-started).count();
        const auto worker_error=error_;
        lock.unlock();
        if(caller_error)std::rethrow_exception(caller_error);
        if(worker_error)std::rethrow_exception(worker_error);
    }
};
}
