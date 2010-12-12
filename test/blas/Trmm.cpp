/*
   Copyright (c) 2009-2010, Jack Poulson
   All rights reserved.

   This file is part of Elemental.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    - Neither the name of the owner nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/
#include <ctime>
#include "elemental.hpp"
#include "elemental/blas_internal.hpp"
using namespace std;
using namespace elemental;
using namespace elemental::wrappers::mpi;

void Usage()
{
    cout << "TRiangular Matrix-Matrix multiplication.\n\n"
         << "  Trmm <r> <c> <side> <shape> <orientation> <unit diag?> <m> <n> "
            "<nb> <print?>\n\n"
         << "  r: number of process rows\n" 
         << "  c: number of process cols\n" 
         << "  side: {L,R}\n"
         << "  shape: {L,U}\n"
         << "  orientation: {N,T,C}\n"
         << "  diag?: {N,U}\n"
         << "  m: height of right-hand sides\n" 
         << "  n: number of right-hand sides\n"
         << "  nb: algorithmic blocksize\n"
         << "  print matrices?: false iff 0\n" << endl;
}

template<typename T>
void TestTrmm
( bool printMatrices, 
  Side side, Shape shape, Orientation orientation, Diagonal diagonal,
  int m, int n, T alpha, const Grid& g )
{
    double startTime, endTime, runTime, gFlops;
    DistMatrix<T,MC,MR> A(g);
    DistMatrix<T,MC,MR> X(g);
    DistMatrix<T,Star,Star> ARef(g);
    DistMatrix<T,Star,Star> XRef(g);

    if( side == Left )
        A.ResizeTo( m, m );
    else
        A.ResizeTo( n, n );
    X.ResizeTo( m, n );

    A.SetToRandom();
    X.SetToRandom();
    if( printMatrices )
    {
        A.Print("A");
        X.Print("X");
    }
    if( g.VCRank() == 0 )
    {
        cout << "  Starting Trmm...";
        cout.flush();
    }
    Barrier( MPI_COMM_WORLD );
    startTime = Time();
    blas::Trmm( side, shape, orientation, diagonal, alpha, A, X );
    Barrier( MPI_COMM_WORLD );
    endTime = Time();
    runTime = endTime - startTime;
    gFlops = blas::internal::TrmmGFlops<T>(side,m,n,runTime);
    if( g.VCRank() == 0 )
    {
        cout << "DONE.\n"
             << "  Time = " << runTime << " seconds. GFlops = " 
             << gFlops << endl;
    }
    if( printMatrices )
        X.Print("X after solve");
}

int main( int argc, char* argv[] )
{
    int rank;
    Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if( argc != 11 )
    {
        if( rank == 0 )
            Usage();
        Finalize();
        return 0;
    }
    try
    {
        int argNum = 0;
        const int r = atoi(argv[++argNum]);
        const int c = atoi(argv[++argNum]);
        const Side side = CharToSide(*argv[++argNum]);
        const Shape shape = CharToShape(*argv[++argNum]);
        const Orientation orientation = CharToOrientation(*argv[++argNum]);
        const Diagonal diagonal = CharToDiagonal(*argv[++argNum]);
        const int m = atoi(argv[++argNum]);
        const int n = atoi(argv[++argNum]);
        const int nb = atoi(argv[++argNum]);
        const bool printMatrices = atoi(argv[++argNum]);
#ifndef RELEASE
        if( rank == 0 )
        {
            cout << "==========================================\n"
                 << " In debug mode! Performance will be poor! \n"
                 << "==========================================" << endl;
        }
#endif
        const Grid g( MPI_COMM_WORLD, r, c );
        SetBlocksize( nb );

        if( rank == 0 )
        {
            cout << "Will test Trmm" << SideToChar(side) 
                                     << ShapeToChar(shape)
                                     << OrientationToChar(orientation) 
                                     << DiagonalToChar(diagonal) << endl;
        }

        if( rank == 0 )
        {
            cout << "---------------------\n"
                 << "Testing with doubles:\n"
                 << "---------------------" << endl;
        }
        TestTrmm<double>
        ( printMatrices, side, shape, orientation, diagonal, 
          m, n, (double)3, g );

#ifndef WITHOUT_COMPLEX
        if( rank == 0 )
        {
            cout << "--------------------------------------\n"
                 << "Testing with double-precision complex:\n"
                 << "--------------------------------------" << endl;
        }
        TestTrmm<dcomplex>
        ( printMatrices, side, shape, orientation, diagonal, 
          m, n, (dcomplex)3, g );
#endif
    }
    catch( exception& e )
    {
#ifndef RELEASE
        DumpCallStack();
#endif
        cerr << "Process " << rank << " caught error message:\n"
             << e.what() << endl;
    }
    Finalize();
    return 0;
}

