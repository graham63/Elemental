/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_HESS_SCHUR_MULTIBULGE_COMPUTE_SHIFTS_HPP
#define EL_HESS_SCHUR_MULTIBULGE_COMPUTE_SHIFTS_HPP

namespace El {
namespace hess_schur {
namespace multibulge {

template<typename Real>
Int ComputeShifts
( const Matrix<Real>& H,
        Matrix<Complex<Real>>& w,
        Int iterBeg,
        Int winBeg,
        Int winEnd,
        Int numShiftsRec,
        Int numIterSinceDeflation,
        Int numStaleIterBeforeExceptional,
  const HessenbergSchurCtrl& ctrlShifts )
{
    DEBUG_CSE
    const Real zero(0);
    const Real exceptShift0(Real(4)/Real(3)),
               exceptShift1(-Real(7)/Real(16));

    const Int shiftBeg = Max(iterBeg,winEnd-numShiftsRec);
    const Int numShifts = winEnd - shiftBeg;
    auto shiftInd = IR(shiftBeg,winEnd);
    auto wShifts = w(shiftInd,ALL); 
    if( numIterSinceDeflation > 0 &&
        Mod(numIterSinceDeflation,numStaleIterBeforeExceptional) == 0 )
    {
        // Use exceptional shifts
        for( Int i=winEnd-1; i>=Max(shiftBeg+1,winBeg+2); i-=2 )
        {
            const Real scale = Abs(H(i,i-1)) + Abs(H(i-1,i-2));
            Real eta00 = exceptShift0*scale + H(i,i);
            Real eta01 = scale;
            Real eta10 = exceptShift1*scale;
            Real eta11 = eta00;
            schur::TwoByTwo
            ( eta00, eta01,
              eta10, eta11,
              w(i-1), w(i) );
        }
        if( shiftBeg == winBeg )
            w(shiftBeg) = w(shiftBeg+1) = H(shiftBeg+1,shiftBeg+1);
    }
    else
    {
        // Compute the eigenvalues of the bottom-right window
        auto HShifts = H(shiftInd,shiftInd);
        auto HShiftsCopy( HShifts );
        HessenbergSchur( HShiftsCopy, wShifts, ctrlShifts );
    }
    if( winBeg-shiftBeg == 2 )
    {
        // Use a single real shift twice instead of using two separate
        // real shifts; we choose the one closest to the bottom-right
        // entry, as it is our best guess as to the smallest eigenvalue
        if( wShifts(numShifts-1).imag() == zero ) 
        {
            if( Abs(wShifts(numShifts-1).real()-H(winEnd-1,winEnd-1)) <
                Abs(wShifts(numShifts-2).real()-H(winEnd-1,winEnd-1)) )
            {
                wShifts(numShifts-2) = wShifts(numShifts-1);
            }
            else
            {
                wShifts(numShifts-1) = wShifts(numShifts-2);
            }
        }
    }
    return shiftBeg;
} 

template<typename Real>
Int ComputeShifts
( const Matrix<Complex<Real>>& H,
        Matrix<Complex<Real>>& w,
        Int iterBeg,
        Int winBeg,
        Int winEnd,
        Int numShiftsRec,
        Int numIterSinceDeflation,
        Int numStaleIterBeforeExceptional,
  const HessenbergSchurCtrl& ctrlShifts )
{
    DEBUG_CSE
    // For some reason, LAPACK suggests only using a single exceptional shift
    // for complex matrices.
    const Real exceptShift0(Real(4)/Real(3));

    const Int shiftBeg = Max(iterBeg,winEnd-numShiftsRec);
    auto shiftInd = IR(shiftBeg,winEnd);
    auto wShifts = w(shiftInd,ALL);
    if( numIterSinceDeflation > 0 &&
        Mod(numIterSinceDeflation,numStaleIterBeforeExceptional) == 0 )
    {
        for( Int i=winEnd-1; i>=shiftBeg+1; i-=2 )
            w(i-1) = w(i) = H(i,i) + exceptShift0*OneAbs(H(i,i-1));
    }
    else
    {
        // Compute the eigenvalues of the bottom-right window
        auto HShifts = H(shiftInd,shiftInd);
        auto HShiftsCopy( HShifts );
        HessenbergSchur( HShiftsCopy, wShifts, ctrlShifts );
    }
    if( winBeg-shiftBeg == 2 )
    {
        // Use the same shift twice; we choose the one closest to the
        // bottom-right entry, as it is our best guess as to the smallest
        // eigenvalue
        if( Abs(w(winEnd-1).real()-H(winEnd-1,winEnd-1)) <
            Abs(w(winEnd-2).real()-H(winEnd-1,winEnd-1)) )
        {
            w(winEnd-2) = w(winEnd-1);
        }
        else
        {
            w(winEnd-1) = w(winEnd-2);
        }
    }
    return shiftBeg;
}

} // namespace multibulge
} // namespace hess_schur
} // namespace El

#endif // ifndef EL_HESS_SCHUR_MULTIBULGE_COMPUTE_SHIFTS_HPP
