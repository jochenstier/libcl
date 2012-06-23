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
#ifndef _oclSobel
#define _oclSobel

#include "oclProgram.h"
#include "oclBuffer.h"
#include "oclImage2D.h"

class oclSobel : public oclProgram
{
    public: 

        oclSobel(oclContext& iContext);

        int compile();

        int compute(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDx, oclImage2D& bfDy);

    protected:

        oclKernel clSobel;
};      

#endif