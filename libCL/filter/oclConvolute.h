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
        
        // apply separable convolution over iAxis to buffer Object
		int conv3D(oclDevice& iDevice, oclBuffer& bfSource, oclBuffer& bfDest, size_t iDim[3], cl_int4 iAxis, oclBuffer& bfFilter);

        // apply separable convolution over iAxis to image
		int conv2D(oclDevice& iDevice, oclImage2D& bfSource, oclImage2D& bfDest, cl_int4 iAxis, oclBuffer& bfFilter);

        // apply 2D kernel to 3D buffer
		//int compute(oclDevice& iDevice, oclBuffer& bfSource, oclBuffer& bfDest, size_t iDim[3], oclImage2D& bfFilter);

    protected:

 		oclKernel clConv2D;
 		oclKernel clConv3D;

        oclBuffer bfGauss3;
        oclBuffer bfGauss5;
        oclBuffer bfLoG;
};      

#endif