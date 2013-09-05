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
#ifndef _oclMemory
#define _oclMemory

#include "oclProgram.h"
#include "oclImage2D.h"
#include "oclBuffer.h"

class oclMemory : public oclProgram
{
    public: 

        oclMemory(oclContext& iContext, oclProgram* iParent = 0);

        static cl_float4 c0000;
        
        int memSet(oclDevice& iDevice, oclImage2D& bfDest, cl_float4 iValue);

        int memSet(oclDevice& iDevice, oclBuffer& bfDest, cl_float4 iValue);
        int memSet(oclDevice& iDevice, oclBuffer& bfDest, cl_float iValue);

        int random(oclDevice& iDevice, oclBuffer& bfDest, cl_float iMin, cl_float iMax);

        int mean(oclDevice& iDevice, oclBuffer& bfSrce, oclBuffer& bfMean);
        int variance(oclDevice& iDevice, oclBuffer& bfSrce, oclBuffer& bfMean, oclBuffer& bfVariance);
        int normalize(oclDevice& iDevice, oclBuffer& bfSrce, oclBuffer& bfDest);

        int min(oclDevice& iDevice, oclBuffer& bfSrce, oclBuffer& bfDest);
        int max(oclDevice& iDevice, oclBuffer& bfSrce, oclBuffer& bfDest);
        int sum(oclDevice& iDevice, oclBuffer& bfSrce, oclBuffer& bfDest);

	protected:

		size_t mMaxWorkgroupSizes[3];

        oclKernel clMin;
        oclKernel clMax;
        oclKernel clMean;
        oclKernel clVariance;
        oclKernel clNormalize;
        //oclKernel clTemp;

        oclKernel clMemSetImage;
        oclKernel clMemSetBuffer_float4;
        oclKernel clMemSetBuffer_float;

		oclBuffer bfMean;
		oclBuffer bfVariance;
};      

#endif