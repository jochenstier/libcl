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
#ifndef _oclImage1D
#define _oclImage1D

#include "oclMem.h"

class oclImage1D : public oclMem
{
    public:

        oclImage1D(oclContext& iContext, char* iName="oclImage1D");

        virtual bool create(cl_mem_flags iMemFlags, cl_image_format& iFormat, size_t iDim0, void* iHostPtr=0);
        virtual bool map(cl_map_flags iMapping, int iDevice = 0);
        virtual bool read(int iDevice = 0);
        virtual bool write(int iDevice = 0);

        //
        virtual bool resize(size_t iDim0, void* iHostPtr=0);

        //
        template <class RESULT> RESULT getImageInfo(cl_image_info iInfo) ;
        // this is a hack because clGetImageInfo does not work interop images
        virtual cl_image_format getImageFormat()
        {
            return getImageInfo<cl_image_format>(CL_IMAGE_FORMAT);
        };

        size_t dim(int iAxis);

    private:

        oclImage1D(const oclImage1D&);
        oclImage1D& operator = (const oclImage1D&);
};

//
//
//

template <class RESULT> RESULT oclImage1D::getImageInfo(cl_image_info iInfo) 
{
    RESULT lResult;
    clGetImageInfo(*this, iInfo, sizeof(RESULT), &lResult, 0);
    return lResult;
}

#endif