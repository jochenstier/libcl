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
#ifndef _oclImage3D
#define _oclImage3D

#include "oclMem.h"

class oclImage3D : public oclMem
{
    public:

        oclImage3D(oclContext& iContext, char* iName="oclImage3D");

        virtual bool create(cl_mem_flags iMemFlags, cl_image_format& iFormat, size_t iDim0, size_t iDim1, size_t iDim2, void* iHostPtr=0);
        virtual bool map(cl_map_flags iMapping, int iDevice = 0);
        virtual bool read(int iDevice = 0);
        virtual bool write(int iDevice = 0);

        //
        virtual bool resize(size_t iDim0, size_t iDim1, size_t iDim2, void* iHostPtr=0);

        //
        template <class RESULT> RESULT getImageInfo(cl_image_info iInfo) ;
        // this is a hack because clGetImageInfo does not work interop images
        virtual cl_image_format getImageFormat()
        {
            return getImageInfo<cl_image_format>(CL_IMAGE_FORMAT);
        };

        size_t dim(int iAxis);

    private:

        oclImage3D(const oclImage3D&);
        oclImage3D& operator = (const oclImage3D&);
};

//
//
//

template <class RESULT> RESULT oclImage3D::getImageInfo(cl_image_info iInfo) 
{
    RESULT lResult;
    clGetImageInfo(*this, iInfo, sizeof(RESULT), &lResult, 0);
    return lResult;
}

#endif