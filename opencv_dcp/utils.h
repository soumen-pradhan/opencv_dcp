#pragma once

/* Fixed Width Types */
#include <cstdint>

using i8 = int8_t;
using u8 = uint8_t;
using i16 = int16_t;
using u16 = uint16_t;
using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;
using f32 = float;
using f64 = double;

/* Cpp Scope Timer */
#include <chrono>
#include <functional>

template <class time_unit = std::chrono::microseconds>
struct Timer {
    using clock = std::chrono::steady_clock;
    //using time_unit = std::chrono::microseconds;
    using handler = std::function<void(time_unit)>;

private:
    clock::time_point start, end;
    handler handle;

public:
    Timer(const handler& func)
        : start(clock::now())
        , handle(func)
    {
    }

    ~Timer()
    {
        end = clock::now();
        auto duration = std::chrono::duration_cast<time_unit>(end - start);
        handle(duration);
    }
};

