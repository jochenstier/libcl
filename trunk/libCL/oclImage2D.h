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
#ifndef _oclImage2D
#define _oclImage2D

#include "oclMem.h"

class oclImage2D : public oclMem
{
    public:

        oclImage2D(oclContext& iContext, char* iName="oclImage2D");
        ~oclImage2D();

        virtual bool create(cl_mem_flags iMemFlags, cl_image_format& iFormat, size_t iDim0, size_t iDim1, void* iHostPtr=0);
        virtual bool map(oclDevice& iDevice, cl_map_flags iMapping);
        virtual bool read(oclDevice& iDevice);
        virtual bool write(oclDevice& iDevice);

        //
        virtual bool resize(size_t iDim0, size_t iDim1, void* iHostPtr=0);

        //
        template <class RESULT> RESULT getImageInfo(cl_image_info iInfo);
        size_t dim(int iAxis);

    private:

        oclImage2D(const oclImage2D&);
        oclImage2D& operator = (const oclImage2D&);

};

//
//
//

template <class RESULT> RESULT oclImage2D::getImageInfo(cl_image_info iInfo) 
{
    RESULT lResult;
    clGetImageInfo(*this, iInfo, sizeof(RESULT), &lResult, 0);
    return lResult;
}


#endif