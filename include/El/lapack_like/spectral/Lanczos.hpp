/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_SPECTRAL_LANCZOS_HPP
#define EL_SPECTRAL_LANCZOS_HPP

namespace El {

// Form
//
//   A V = V T + v (beta e_{k-1})^H.
//
// T is returned instead of just the tridiagonal portion since later versions
// may account for T numerically having an upper-Hessenberg perturbation.
//
// The current version was implemented for the purposes of gaining rough
// estimates of the extremal singular values in order to keep the condition
// number of the augmented system
//
//      | alpha*I  A | | r/alpha | = | b |
//      |   A^H    0 | |    x    |   | 0 |
//
// as near to that of A as possible (e.g., by setting alpha to roughly the
// minimum nonzero singular value of A), as well as to provide a means of
// scaling A down to roughly unit two-norm.
//

template<typename Field,class ApplyAType>
void Lanczos
(       Int n,
  const ApplyAType& applyA,
        Matrix<Base<Field>>& T,
        Int basisSize )
{
    EL_DEBUG_CSE
    typedef Base<Field> Real;
    const Real eps = limits::Epsilon<Real>();

    Matrix<Field> v_km1, v_k, v;
    basisSize = Min(n,basisSize);
    Zeros( v_km1, n, 1 );
    Zeros( v_k,   n, 1 );
    Zeros( T, basisSize, basisSize );

    // Create the initial unit-vector
    // ------------------------------
    {
        Uniform( v, n, 1 );
        Shift( v, SampleUniform<Field>() );
        const Real beta = FrobeniusNorm( v );
        v *= 1/beta;
    }

    // TODO(poulson): Incorporate Frobenius norm of A?
    const Real minBeta = eps;
    for( Int k=0; k<basisSize; ++k )
    {
        if( k > 0 )
            v_km1 = v_k;
        v_k = v;

        // w := A v
        // --------
        applyA( v_k, v );

        // w := w - T(k-1,k) v_{k-1}
        // -------------------------
        if( k > 0 )
            Axpy( -T(k-1,k), v_km1, v );

        // w := w - T(k,k) v_k
        // -------------------
        const Real tau = RealPart(Dot(v_k,v));
        T(k,k) = tau;
        Axpy( -tau, v_k, v );

        // v := w / || w ||_2
        // -----------------
        const Real beta = FrobeniusNorm( v );
        if( beta <= minBeta )
        {
            T.Resize( k+1, k+1 );
            break;
        }
        v *= 1/beta;

        // Expand the Rayleigh quotient as appropriate
        // -------------------------------------------
        if( k < basisSize-1 )
        {
            T(k+1,k) = T(k,k+1) = beta;
        }
    }
}

template<typename Field,class ApplyAType>
Base<Field> LanczosDecomp
(       Int n,
  const ApplyAType& applyA,
        Matrix<Field>& V,
        Matrix<Base<Field>>& T,
        Matrix<Field>& v,
        Int basisSize )
{
    EL_DEBUG_CSE
    typedef Base<Field> Real;
    const Real eps = limits::Epsilon<Real>();

    basisSize = Min(n,basisSize);
    Zeros( V, n, basisSize );
    Zeros( T, basisSize, basisSize );

    // Initialize the first column of V
    // --------------------------------
    {
        Uniform( v, n, 1 );
        Shift( v, SampleUniform<Field>() );
        const Real beta = FrobeniusNorm( v );
        v *= 1/beta;
    }
    if( basisSize > 0 )
    {
        auto v0 = V( ALL, IR(0) );
        v0 = v;
    }

    Real beta = 0;
    // TODO(poulson): Incorporate Frobenius norm of A
    const Real minBeta = eps;
    for( Int k=0; k<basisSize; ++k )
    {
        // w := A^H (A v)
        // --------------
        auto v_k  = V( ALL, IR(k) );
        applyA( v_k, v );

        // w := w - T(k-1,k) v_{k-1}
        // -------------------------
        if( k > 0 )
        {
            auto v_km1 = V( ALL, IR(k-1) );
            Axpy( -T(k-1,k), v_km1, v );
        }

        // w := w - T(k,k) v_k
        // -------------------
        const Real tau = RealPart(Dot(v_k,v));
        T(k,k) = tau;
        Axpy( -tau, v_k, v );

        // v := w / || w ||_2
        // -----------------
        beta = FrobeniusNorm( v );
        if( beta <= minBeta )
        {
            T.Resize( k+1, k+1 );
            break;
        }
        v *= 1/beta;

        // Expand the Lanczos decomposition as appropriate
        // -----------------------------------------------
        if( k < basisSize-1 )
        {
            T(k+1,k) = T(k,k+1) = beta;
            auto v_kp1 = V( ALL, IR(k+1) );
            v_kp1 = v;
        }
    }
    return beta;
}

} // namespace El

#endif // ifndef EL_SPECTRAL_LANCZOS
