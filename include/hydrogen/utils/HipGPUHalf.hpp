#ifndef HYDROGEN_UTILS_HIPGPUHALF_HPP_
#define HYDROGEN_UTILS_HIPGPUHALF_HPP_

#include <hydrogen/meta/MetaUtilities.hpp>
#include <hydrogen/device/GPU.hpp>

#include <hip/hip_fp16.h>

namespace hydrogen
{
class HipGPUHalf
{
public:

  template <typename T, EnableWhen<std::is_integral<T>, int> = 0>
  HipGPUHalf(T val)
    : val_{val}
  {}
  template <typename T, EnableWhen<std::is_floating_point<T>, int> = 0>
  HipGPUHalf(T val)
    : val_{val}
  {}

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
  inline __half get_val() { return float(val_); }

  __half val_;

};// class HipGPUHalf

// +=
template <typename T>
T& operator+=(T& lhs, HipGPUHalf const& rhs)
{
  lhs =  float(lhs) + float(rhs.val_);

  return lhs;
}

// -=
template <typename T>
T& operator-=(T& lhs, HipGPUHalf const& rhs)
{
  lhs =  float(lhs) - float(rhs.val_);

  return lhs;
}

// *=
template <typename T>
T& operator*=(T& lhs, HipGPUHalf const& rhs)
{
  lhs =  float(lhs) * float(rhs.val_);

  return lhs;
}

// /=
template <typename T>
T& operator/=(T& lhs, HipGPUHalf const& rhs)
{
  lhs =  float(lhs) / float(rhs.val_);

  return lhs;
}

/** @brief Enable addition functionality for __half on GPUs. */
HipGPUHalf operator+(HipGPUHalf const& lhs, HipGPUHalf const& rhs)
{
  return float(lhs.val_) + float(rhs.val_);
}
template<typename T>
HipGPUHalf operator+(T const& lhs, HipGPUHalf const& rhs)
{
  return float(lhs) + float(rhs.val_);
}
template<typename T>
HipGPUHalf operator+(HipGPUHalf const& lhs, T const& rhs)
{
  return float(lhs.val_) + float(rhs);
}

/** @brief Enable subtraction functionality for __half on GPUs. */
HipGPUHalf operator-(HipGPUHalf const& lhs, HipGPUHalf const& rhs)
{
  return float(lhs.val_) - float(rhs.val_);
}
template<typename T>
HipGPUHalf operator-(T const& lhs, HipGPUHalf const& rhs)
{
  return float(lhs) - float(rhs.val_);
}
template<typename T>
HipGPUHalf operator-(HipGPUHalf const& lhs, T const& rhs)
{
  return float(lhs.val_) - float(rhs);
}

/** @brief Enable multiplication functionality for __half on GPUs. */
HipGPUHalf operator*(HipGPUHalf const& lhs, HipGPUHalf const& rhs)
{
  return float(lhs.val_) * float(rhs.val_);
}
template<typename T>
HipGPUHalf operator*(T const& lhs, HipGPUHalf const& rhs)
{
  return float(lhs) * float(rhs.val_);
}
template<typename T>
HipGPUHalf operator*(HipGPUHalf const& lhs, T const& rhs)
{
  return float(lhs.val_) * float(rhs);
}

/** @brief Enable division functionality for __half on GPUs. */
HipGPUHalf operator/(HipGPUHalf const& lhs, HipGPUHalf const& rhs)
{
  return float(lhs.val_) / float(rhs.val_);
}

template<typename T>
HipGPUHalf operator/(T const& lhs, HipGPUHalf const& rhs)
{
  return float(lhs) / float(rhs.val_);
}
template<typename T>
HipGPUHalf operator/(HipGPUHalf const& lhs, T const& rhs)
{
  return float(lhs.val_) / float(rhs);
}

/** @brief Enable unary minus functionality for __half. */
HipGPUHalf operator-(HipGPUHalf const& x)
{
  return -float(x.val_);
}

/** @brief Enable equality functionality for __half. */
bool operator==(HipGPUHalf const& rhs, HipGPUHalf const& lhs)
{
  return float(rhs.val_) == float(lhs.val_);
}
template <typename T>
bool operator==(T const& rhs, HipGPUHalf const& lhs)
{
  float(rhs) == float(lhs.val_);
}
template <typename T>
bool operator==(HipGPUHalf const& rhs, T const& lhs)
{
  float(rhs.val_) == float(lhs);
}

/** @brief Enable ostream functionality for __half. */
std::ostream& operator<<(std::ostream& os, HipGPUHalf& const x)
{
  return os << float(x.val_) << "_h";
}


}// namespace hydrogen
#endif // HYDROGEN_SYNCINFOBASE_HPP_
