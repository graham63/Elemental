#include <catch2/catch.hpp>

#include <hydrogen/utils/HalfPrecision.hpp>
#include <hydrogen/utils/HipGPUHalf.hpp>

#include <hip/hip_fp16.h>

#include <sstream>

using gpu_half_type = hydrogen::HipGPUHalf;

// Test static properties here. Just use static_assert, since it's
// generally sufficient.

// Construct by integral types
static_assert(std::is_constructible<gpu_half_type, int>::value,
              "gpu_half_type should be constructible by an int.");
static_assert(std::is_constructible<gpu_half_type, long>::value,
              "gpu_half_type should be constructible by an long.");
static_assert(std::is_constructible<gpu_half_type, size_t>::value,
              "gpu_half_type should be constructible by an size_t.");
// Construct by IEEE-754 floating point types
static_assert(std::is_constructible<gpu_half_type, float>::value,
              "gpu_half_type should be constructible by a float.");
static_assert(std::is_constructible<gpu_half_type, double>::value,
              "gpu_half_type should be constructible by a double.");

// Runtime testing
using namespace hydrogen;
TEST_CASE("Testing HipGPUHalf operations", "[seq][gpu][half][hip]")
{
  SECTION("Testing operator + ")
  {
    HipGPUHalf one_h(1.f);
    HipGPUHalf two_h(2.f);
    HipGPUHalf three_h(3.f);

    CHECK( one_h + two_h == three_h );
    CHECK( two_h + one_h == three_h );
    CHECK_FALSE( one_h + one_h == three_h );

    float one_f(1);

    CHECK( one_f + one_h == two_h );
    CHECK( one_h + one_f == two_h );
    CHECK_FALSE( one_f + one_h == three_h );

    double one_d(1);

    CHECK( one_d + one_h == two_h );
    CHECK( one_h + one_d == two_h );
    CHECK_FALSE( one_d + one_h == three_h );
  }

  SECTION("Testing operator - ")
  {
    gpu_half_type one_h(1.f);
    gpu_half_type neg_one_h(-1.f);
    gpu_half_type two_h(2.f);
    gpu_half_type three_h(3.f);

    CHECK( three_h - two_h == one_h );
    CHECK( two_h - three_h == neg_one_h );
    CHECK_FALSE( one_h - one_h == three_h );

    float two_f(2);

    CHECK( three_h - two_f == one_h );
    CHECK( two_f - three_h == neg_one_h );
    CHECK_FALSE( one_h - one_h == three_h );

    double two_d(2);

    CHECK( three_h - two_d == one_h );
    CHECK( two_d - three_h == neg_one_h );
    CHECK_FALSE( one_h - one_h == three_h );
  }

  SECTION("Testing operator * ")
  {
    gpu_half_type one_h(1.f);
    gpu_half_type two_h(2.f);

    CHECK( one_h * two_h == two_h );
    CHECK( two_h * one_h == two_h );
    CHECK_FALSE( one_h * one_h == two_h );

    float one_f(1);

    CHECK( one_f * two_h == two_h );
    CHECK( two_h * one_f == two_h );
    CHECK_FALSE( one_f * one_h == two_h );

    double one_d(1);

    CHECK( one_d * two_h == two_h );
    CHECK( two_h * one_d == two_h );
    CHECK_FALSE( one_d * one_h == two_h );
  }

  SECTION("Testing operator / ")
  {
    gpu_half_type one_h(1.f);
    gpu_half_type two_h(2.f);

    CHECK( two_h / two_h == one_h );
    CHECK( two_h / one_h == two_h );
    CHECK_FALSE( one_h / one_h == two_h );

    float two_f(2);

    CHECK( two_f / two_h == one_h );
    CHECK( two_h / two_f == one_h );
    CHECK( two_f / one_h == two_h );
    CHECK_FALSE( two_f / two_h == two_h );

    double two_d(2);

    CHECK( two_d / two_h == one_h );
    CHECK( two_h / two_d == one_h );
    CHECK( two_d / one_h == two_h );
    CHECK_FALSE( two_d / two_h == two_h );
  }

  SECTION("Testing operator += ")
  {
    gpu_half_type one_h(1.f);
    gpu_half_type two_h(2.f);
    gpu_half_type three_h(3.f);

    one_h += two_h;
    CHECK( one_h == three_h );
    one_h += two_h;
    CHECK_FALSE( one_h == three_h );

    float one_f(1);

    two_h += one_f;
    CHECK( two_h == three_h );
    two_h += one_f;
    CHECK_FALSE( two_h == three_h );

    float two_f(2);
    float five_f(5);

    two_f += three_h;
    CHECK( two_f == five_f );
    two_f += three_h;
    CHECK_FALSE( two_f == five_f );

    double one_d(1);
    gpu_half_type four_h(4.f);

    three_h += one_d;
    CHECK( three_h == four_h );
    three_h += one_d;
    CHECK_FALSE( three_h == four_h );

    double two_d(2);
    double six_d(6);

    two_d += four_h;
    CHECK( two_d == six_d );
    two_d += four_h;
    CHECK_FALSE( two_d == six_d );
  }

  SECTION("Testing operator -= ")
  {
    gpu_half_type one_h(1.f);
    gpu_half_type two_h(2.f);
    gpu_half_type three_h(3.f);

    three_h -= two_h;
    CHECK( one_h == three_h );
    three_h -= two_h;
    CHECK_FALSE( one_h == three_h );

    float one_f(1);

    two_h -= one_f;
    CHECK( two_h == one_h );
    two_h -= one_f;
    CHECK_FALSE( two_h == one_h );

    float two_f(2);

    two_f -= one_h;
    CHECK( two_f == one_f );
    two_f -= one_h;
    CHECK_FALSE( two_f == one_f );

    double three_d(3);
    gpu_half_type four_h(4.f);

    four_h -= three_d;
    CHECK( four_h == one_h );
    four_h -= three_d;
    CHECK_FALSE( four_h == one_h );

    double two_d(2);
    double one_d(1);

    two_d -= one_h;
    CHECK( two_d == one_d );
    two_d -= one_h;
    CHECK_FALSE( two_d == one_d );
  }

  SECTION("Testing operator *= ")
  {
    gpu_half_type two_h(2.f);
    gpu_half_type three_h(3.f);
    gpu_half_type six_h(6.f);

    two_h *= three_h;
    CHECK( two_h == six_h );
    two_h *= three_h;
    CHECK_FALSE( two_h == six_h );

    float two_f(2);

    three_h *= two_f;
    CHECK( three_h == six_h );
    three_h *= two_f;
    CHECK_FALSE( three_h == six_h );

    gpu_half_type four_h(4.f);
    float eight_f(8);

    two_f *= four_h;
    CHECK( two_f == eight_f );
    two_f *= four_h;
    CHECK_FALSE( two_f == eight_f );

    gpu_half_type eight_h(8.f);
    double two_d(2);

    four_h *= two_d;
    CHECK( four_h == eight_h );
    four_h *= two_d;
    CHECK_FALSE( four_h == eight_h );

    gpu_half_type five_h(5.f);
    double ten_d(10);

    two_d *= five_h;
    CHECK( two_d == ten_d );
    two_d *= five_h;
    CHECK_FALSE( two_d == ten_d );
  }

  SECTION("Testing operator /= ")
  {
    gpu_half_type two_h(2.f);
    gpu_half_type three_h(3.f);
    gpu_half_type six_h(6.f);

    six_h /= two_h;
    CHECK( six_h == three_h );
    six_h /= two_h;
    CHECK_FALSE( six_h == three_h );

    gpu_half_type one_h(1.f);
    float two_f(2);

    two_h /= two_f;
    CHECK( two_h == one_h );
    two_h /= two_f;
    CHECK_FALSE( two_h == one_h );

    float nine_f(9);
    float three_f(3);

    nine_f /= three_h;
    CHECK( nine_f == three_f );
    nine_f /= three_h;
    CHECK_FALSE( nine_f == three_f );

    gpu_half_type four_h(4.f);
    gpu_half_type eight_h(8.f);
    double two_d(2);

    eight_h /= two_d;
    CHECK( eight_h == four_h );
    eight_h /= two_d;
    CHECK_FALSE( eight_h == four_h );

    double eight_d(8);

    eight_d /= four_h;
    CHECK( eight_d == two_d );
    eight_d /= four_h;
    CHECK_FALSE( eight_d == two_d );
  }

  SECTION("Testing operator == ")
  {
    gpu_half_type one(1.f);
    gpu_half_type one_again(1.f);
    gpu_half_type two(2.f);

    CHECK( one == one );
    CHECK( one == one_again );
    CHECK( one_again == one );
    CHECK_FALSE( one == two );
  }

  SECTION("Testing unary minus operator ")
  {
    gpu_half_type one_h(1.f);
    gpu_half_type neg_one_h(-1.f);

    CHECK( -(one_h) == neg_one_h );
  }
}
