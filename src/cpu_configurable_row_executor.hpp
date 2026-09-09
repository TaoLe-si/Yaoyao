#pragma once
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>
#include <stdexcept>
#include <cstdint>
namespace tao::dual {
// Caller participates. Only 2 or 4 total threads, no spin or per-task allocation.
// run is synchronous, nonrecursive; destruction must not race public calls.
class CpuConfigurableRowExecutor {
    const size_t threads_, threshold_;
    std::mutex callers_, mutex_;
    std::condition_variable ready_, done_;
    bool stop_=false;
    uint64_t generation_=0;
    size_t remaining_=0, rows_=0;
    void* context_=nullptr;
    void (*invoke_)(void*,size_t,size_t)=nullptr;
    std::exception_ptr error_;
    std::vector<std::thread> workers_;
    size_t boundary(size_t rows,size_t i)const {
        return (rows/threads_)*i+((rows%threads_)*i)/threads_;
    }
    void loop(size_t index){
        uint64_t seen=0;
        std::unique_lock<std::mutex> lock(mutex_);
        for(;;){
            ready_.wait(lock,[&]{return stop_||generation_!=seen;});
            if(stop_)return;
            seen=generation_;auto fn=invoke_;auto ctx=context_;
            const size_t begin=boundary(rows_,index),end=boundary(rows_,index+1);
            lock.unlock();std::exception_ptr error;
            try{fn(ctx,begin,end);}catch(...){error=std::current_exception();}
            lock.lock();if(error&&!error_)error_=error;
            if(--remaining_==0)done_.notify_one();
        }
    }
    void shutdown(){
        {std::lock_guard<std::mutex> lock(mutex_);stop_=true;}
        ready_.notify_all();
        for(auto&t:workers_)if(t.joinable())t.join();
    }
public:
    explicit CpuConfigurableRowExecutor(size_t threads=2,size_t threshold=262144)
        :threads_(threads),threshold_(threshold){
        if((threads!=2&&threads!=4)||!threshold)throw std::invalid_argument("executor configuration");
        workers_.reserve(threads-1);
        try{for(size_t i=1;i<threads;++i)workers_.emplace_back([this,i]{loop(i);});}
        catch(...){shutdown();throw;}
    }
    CpuConfigurableRowExecutor(const CpuConfigurableRowExecutor&)=delete;
    CpuConfigurableRowExecutor& operator=(const CpuConfigurableRowExecutor&)=delete;
    ~CpuConfigurableRowExecutor(){shutdown();}
    template<class F>void run(size_t rows,size_t cols,F fn){
        std::lock_guard<std::mutex> caller(callers_);
        if(rows<threads_||cols==0||rows<threshold_/cols+(threshold_%cols!=0)){
            fn(0,rows);return;
        }
        {std::lock_guard<std::mutex> lock(mutex_);
            rows_=rows;context_=&fn;error_=nullptr;remaining_=threads_-1;
            invoke_=[](void*p,size_t b,size_t e){(*static_cast<F*>(p))(b,e);};
            ++generation_;
        }
        ready_.notify_all();std::exception_ptr caller_error;
        try{fn(0,boundary(rows,1));}catch(...){caller_error=std::current_exception();}
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock,[&]{return remaining_==0;});
        auto worker_error=error_;context_=nullptr;invoke_=nullptr;lock.unlock();
        if(caller_error)std::rethrow_exception(caller_error);
        if(worker_error)std::rethrow_exception(worker_error);
    }
};
}
