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
#ifndef _oclBilateral
#define _oclBilateral

#include "oclProgram.h"
#include "oclBuffer.h"
#include "oclImage2D.h"

class oclBilateral : public oclProgram
{
    public: 

        oclBilateral(oclContext& iContext);

        int compile();

        int iso2D(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, cl_int iRadius, cl_float iRange, cl_float4 iMask);
        int aniso2Dtang(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, cl_int iRadius, cl_float iRange, oclImage2D& bfVector, cl_float4 iMask);
        int aniso2Dorth(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, cl_int iRadius, cl_float iRange, oclImage2D& bfVector, cl_float4 iMask);

    protected:

        oclKernel clIso2D;
        oclKernel clAniso2Dtang;
        oclKernel clAniso2Dorth;
};      

#endif