#ifndef HYDROGEN_UTILS_HIPGPUHALF_HPP_
#define HYDROGEN_UTILS_HIPGPUHALF_HPP_

#if defined __HIPCC__
#define H_HOSTDEVICE __host__ __device__
#else
#define H_HOSTDEVICE
#endif

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

    H_HOSTDEVICE HipGPUHalf()
        : val_{0.f}
    {}

    template <typename T, EnableWhen<std::is_integral<T>, int> = 0>
    H_HOSTDEVICE HipGPUHalf(T val)
        : val_{val}
    {}
    template <typename T, EnableWhen<std::is_floating_point<T>, int> = 0>
    H_HOSTDEVICE HipGPUHalf(T val)
        : val_{val}
    {}

    template <typename T, EnableWhen<Or<std::is_integral<T>,
                                        std::is_floating_point<T>>, int> = 0>
    H_HOSTDEVICE operator T() const noexcept
    {
        return val_;
    }
    H_HOSTDEVICE operator double() const noexcept
    {
        return float(val_);
    }
    operator rocblas_half() const noexcept
    {
        return *reinterpret_cast<rocblas_half const*>(&val_);;
    }

    /** @brief Enable addition assignment for __half on GPUs. */
    H_HOSTDEVICE inline HipGPUHalf& operator+=(HipGPUHalf const& rhs)
    {
        val_ = float(val_) + float(rhs.val_);
        return *this;
    }

    /** @brief Enable subtraction assignment for __half on GPUs. */
    H_HOSTDEVICE inline HipGPUHalf& operator-=(HipGPUHalf const& rhs)
    {
        val_ = float(val_) - float(rhs.val_);
        return *this;
    }

    /** @brief Enable multiplication assignment for __half on GPUs. */
    H_HOSTDEVICE inline HipGPUHalf& operator*=(HipGPUHalf const& rhs)
    {
        val_ = float(val_) * float(rhs.val_);
        return *this;
    }

    /** @brief Enable division assignment for __half on GPUs. */
    H_HOSTDEVICE inline HipGPUHalf& operator/=(HipGPUHalf const& rhs)
    {
        val_ = float(val_) / float(rhs.val_);
        return *this;
    }

};// class HipGPUHalf


/** @brief Enable addition functionality for __half on GPUs. */
H_HOSTDEVICE HipGPUHalf operator+(HipGPUHalf const& lhs, HipGPUHalf const& rhs)
{
    return float(lhs) + float(rhs);
}
template<typename T>
H_HOSTDEVICE HipGPUHalf operator+(T const& lhs, HipGPUHalf const& rhs)
{
    return lhs + float(rhs);
}
template<typename T>
H_HOSTDEVICE HipGPUHalf operator+(HipGPUHalf const& lhs, T const& rhs)
{
    return float(lhs) + rhs;
}

/** @brief Enable subtraction functionality for __half on GPUs. */
H_HOSTDEVICE HipGPUHalf operator-(HipGPUHalf const& lhs, HipGPUHalf const& rhs)
{
    return float(lhs) - float(rhs);
}
template<typename T>
H_HOSTDEVICE HipGPUHalf operator-(T const& lhs, HipGPUHalf const& rhs)
{
    return lhs - float(rhs);
}
template<typename T>
H_HOSTDEVICE HipGPUHalf operator-(HipGPUHalf const& lhs, T const& rhs)
{
    return float(lhs) - rhs;
}

/** @brief Enable multiplication functionality for __half on GPUs. */
H_HOSTDEVICE HipGPUHalf operator*(HipGPUHalf const& lhs, HipGPUHalf const& rhs)
{
    return float(lhs) * float(rhs);
}
template<typename T>
H_HOSTDEVICE HipGPUHalf operator*(T const& lhs, HipGPUHalf const& rhs)
{
    return lhs * float(rhs);
}
template<typename T>
H_HOSTDEVICE HipGPUHalf operator*(HipGPUHalf const& lhs, T const& rhs)
{
    return float(lhs) * rhs;
}

/** @brief Enable division functionality for __half on GPUs. */
H_HOSTDEVICE HipGPUHalf operator/(HipGPUHalf const& lhs, HipGPUHalf const& rhs)
{
    return float(lhs) / float(rhs);
}

template<typename T>
H_HOSTDEVICE HipGPUHalf operator/(T const& lhs, HipGPUHalf const& rhs)
{
    return lhs / float(rhs);
}
template<typename T>
H_HOSTDEVICE HipGPUHalf operator/(HipGPUHalf const& lhs, T const& rhs)
{
    return float(lhs) / rhs;
}

// +=
template <typename T>
H_HOSTDEVICE T& operator+=(T& lhs, HipGPUHalf const& rhs)
{
    lhs += float(rhs);
    return lhs;
}

// -=
template <typename T>
H_HOSTDEVICE T& operator-=(T& lhs, HipGPUHalf const& rhs)
{
    lhs -= float(rhs);
    return lhs;
}

// *=
template <typename T>
H_HOSTDEVICE T& operator*=(T& lhs, HipGPUHalf const& rhs)
{
    lhs *= float(rhs);
    return lhs;
}

// /=
template <typename T>
H_HOSTDEVICE T& operator/=(T& lhs, HipGPUHalf const& rhs)
{
    lhs /= float(rhs);
    return lhs;
}

// NEGATIVE
H_HOSTDEVICE HipGPUHalf operator-(HipGPUHalf const& x)
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

bool operator<=(HipGPUHalf const& lhs, HipGPUHalf const& rhs)
{
    return float(lhs) <= float(rhs);
}
template <typename T>
bool operator<=(T const& lhs, HipGPUHalf const& rhs)
{
    return lhs <= float(rhs);
}
template <typename T>
bool operator<=(HipGPUHalf const& lhs, T const& rhs)
{
    return float(lhs) <= rhs;
}

bool operator>=(HipGPUHalf const& lhs, HipGPUHalf const& rhs)
{
    return float(lhs) >= float(rhs);
}
template <typename T>
bool operator>=(T const& lhs, HipGPUHalf const& rhs)
{
    return lhs >= float(rhs);
}
template <typename T>
bool operator>=(HipGPUHalf const& lhs, T const& rhs)
{
    return float(lhs) >= rhs;
}

inline std::ostream& operator<<(std::ostream& os, HipGPUHalf const& x)
{
    return os << float(x) << "_h";
}

}// namespace hydrogen

#endif // HYDROGEN_SYNCINFOBASE_HPP_
