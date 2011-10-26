// Copyright [2011] [Geist Software Labs Inc.]
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef _oclConvolute
#define _oclConvolute

#include "oclProgram.h"
#include "oclBuffer.h"
#include "oclImage2D.h"

class oclConvolute : public oclProgram
{
    public: 

	    oclConvolute(oclContext& iContext);

		int compile();

        // 2D convolutions on images and buffers
		int iso2D(oclDevice& iDevice, oclImage2D& bfSource, oclImage2D& bfDest, oclBuffer& bfFilter, int iFilterW, int iFilterH);
		int iso2Dsep(oclDevice& iDevice, oclImage2D& bfSource, oclImage2D& bfDest, cl_int2 iAxis, oclBuffer& bfFilter);
		int aniso2Dtang(oclDevice& iDevice, oclImage2D& bfSource, oclImage2D& bfDest, oclImage2D& iLine, oclBuffer& bfFilter);
		int aniso2Dorth(oclDevice& iDevice, oclImage2D& bfSource, oclImage2D& bfDest, oclImage2D& iLine, oclBuffer& bfFilter);

        // create kernels
        static bool gauss1D(float iSigma, oclBuffer& iBuffer);
        static bool gauss2D(float iSigma, oclBuffer& iBuffer, int iFilterW, int iFilterH);

        static bool DoG2D(float iSigmaA, float iSigmaB, float iSensitivity, oclBuffer& iBuffer, int iKernelW, int iKernelH);
        static bool DoG1D(float iSigmaA, float iSigmaB, float iSensitivity, oclBuffer& iBuffer);

    protected:

 		oclKernel clIso2D;
 		oclKernel clIso2Dsep;
 		oclKernel clAniso2Dtang;
 		oclKernel clAniso2Dorth;
};      

#endif