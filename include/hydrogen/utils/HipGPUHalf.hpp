#ifndef HYDROGEN_UTILS_HIPGPUHALF_HPP_
#define HYDROGEN_UTILS_HIPGPUHALF_HPP_

#include <hydrogen/meta/MetaUtilities.hpp>

#include <hip/hip_fp16.h>
#define ROCM_USE_FLOAT16
#include <rocblas.h>

#include <iostream>

namespace hydrogen
{

class HipGPUHalf
{
private:

    __half val_;

public:

    HipGPUHalf()
        : val_{0.f}
    {}

    template <typename T, EnableWhen<std::is_integral<T>, int> = 0>
    HipGPUHalf(T val)
        : val_{val}
    {}
    template <typename T, EnableWhen<std::is_floating_point<T>, int> = 0>
    HipGPUHalf(T val)
        : val_{val}
    {}

    template <typename T, EnableWhen<Or<std::is_integral<T>,
                                        std::is_floating_point<T>>, int> = 0>
    operator T() const noexcept
    {
        return val_;
    }
    operator double() const noexcept
    {
        return float(val_);
    }
    operator rocblas_half() const noexcept
    {
        return val_;
    }

    /** @brief Enable addition assignment for __half on GPUs. */
    inline HipGPUHalf& operator+=(HipGPUHalf const& rhs)
    {
        val_ = float(val_) + float(rhs.val_);
        return *this;
    }

    /** @brief Enable subtraction assignment for __half on GPUs. */
    inline HipGPUHalf& operator-=(HipGPUHalf const& rhs)
    {
        val_ = float(val_) - float(rhs.val_);
        return *this;
    }

    /** @brief Enable multiplication assignment for __half on GPUs. */
    inline HipGPUHalf& operator*=(HipGPUHalf const& rhs)
    {
        val_ = float(val_) * float(rhs.val_);
        return *this;
    }

    /** @brief Enable division assignment for __half on GPUs. */
    inline HipGPUHalf& operator/=(HipGPUHalf const& rhs)
    {
        val_ = float(val_) / float(rhs.val_);
        return *this;
    }

};// class HipGPUHalf


/** @brief Enable addition functionality for __half on GPUs. */
HipGPUHalf operator+(HipGPUHalf const& lhs, HipGPUHalf const& rhs)
{
    return float(lhs) + float(rhs);
}
template<typename T>
HipGPUHalf operator+(T const& lhs, HipGPUHalf const& rhs)
{
    return lhs + float(rhs);
}
template<typename T>
HipGPUHalf operator+(HipGPUHalf const& lhs, T const& rhs)
{
    return float(lhs) + rhs;
}

/** @brief Enable subtraction functionality for __half on GPUs. */
HipGPUHalf operator-(HipGPUHalf const& lhs, HipGPUHalf const& rhs)
{
    return float(lhs) - float(rhs);
}
template<typename T>
HipGPUHalf operator-(T const& lhs, HipGPUHalf const& rhs)
{
    return lhs - float(rhs);
}
template<typename T>
HipGPUHalf operator-(HipGPUHalf const& lhs, T const& rhs)
{
    return float(lhs) - rhs;
}

/** @brief Enable multiplication functionality for __half on GPUs. */
HipGPUHalf operator*(HipGPUHalf const& lhs, HipGPUHalf const& rhs)
{
    return float(lhs) * float(rhs);
}
template<typename T>
HipGPUHalf operator*(T const& lhs, HipGPUHalf const& rhs)
{
    return lhs * float(rhs);
}
template<typename T>
HipGPUHalf operator*(HipGPUHalf const& lhs, T const& rhs)
{
    return float(lhs) * rhs;
}

/** @brief Enable division functionality for __half on GPUs. */
HipGPUHalf operator/(HipGPUHalf const& lhs, HipGPUHalf const& rhs)
{
    return float(lhs) / float(rhs);
}

template<typename T>
HipGPUHalf operator/(T const& lhs, HipGPUHalf const& rhs)
{
    return lhs / float(rhs);
}
template<typename T>
HipGPUHalf operator/(HipGPUHalf const& lhs, T const& rhs)
{
    return float(lhs) / rhs;
}

// +=
template <typename T>
T& operator+=(T& lhs, HipGPUHalf const& rhs)
{
    lhs += float(rhs);
    return lhs;
}

// -=
template <typename T>
T& operator-=(T& lhs, HipGPUHalf const& rhs)
{
    lhs -= float(rhs);
    return lhs;
}

// *=
template <typename T>
T& operator*=(T& lhs, HipGPUHalf const& rhs)
{
    lhs *= float(rhs);
    return lhs;
}

// /=
template <typename T>
T& operator/=(T& lhs, HipGPUHalf const& rhs)
{
    lhs /= float(rhs);
    return lhs;
}

// NEGATIVE
HipGPUHalf operator-(HipGPUHalf const& x)
{
    return -float(x);
}

// EQUALITY
bool operator==(HipGPUHalf const& lhs, HipGPUHalf const& rhs)
{
    return float(lhs) == float(rhs);
}
template <typename T>
bool operator==(T const& lhs, HipGPUHalf const& rhs)
{
    return lhs == float(rhs);
}
template <typename T>
bool operator==(HipGPUHalf const& lhs, T const& rhs)
{
    return float(lhs) == rhs;
}

bool operator!=(HipGPUHalf const& lhs, HipGPUHalf const& rhs)
{
    return !(float(lhs) == float(rhs));
}
template <typename T>
bool operator!=(T const& lhs, HipGPUHalf const& rhs)
{
    return !(lhs == float(rhs));
}
template <typename T>
bool operator!=(HipGPUHalf const& lhs, T const& rhs)
{
    return !(float(lhs) == rhs);
}

}// namespace hydrogen

#endif // HYDROGEN_SYNCINFOBASE_HPP_
