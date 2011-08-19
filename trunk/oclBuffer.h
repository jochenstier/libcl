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
#ifndef _oclBuffer
#define _oclBuffer

#include "oclMem.h"

class oclBuffer : public oclMem
{
    public:

        oclBuffer(oclContext& iContext, char* iName="oclBuffer");
        ~oclBuffer();

        //
        template <class TYPE> bool create(cl_mem_flags iMemFlags, size_t iElements, void* iHostPtr=0);
        virtual bool map(oclDevice& iDevice, cl_map_flags iMapping);
        virtual bool read(oclDevice& iDevice);
        virtual bool write(oclDevice& iDevice);

        //
        template <class TYPE> bool resize(size_t iElements, void* iHostPtr=0) ;
        virtual bool resize(size_t iSize, void* iHostPtr = 0);

        //
        template <class TYPE> size_t count();
        virtual size_t dim(int iAxis);

    private:

        oclBuffer(const oclBuffer&);
        oclBuffer& operator = (const oclBuffer&);
};

//
//
//

template <class TYPE> bool oclBuffer::create(cl_mem_flags iMemFlags, size_t iElements, void* iHostPtr)
{
    mMemPtr = clCreateBuffer(mContext, iMemFlags, iElements*sizeof(TYPE), iHostPtr, &sStatusCL);
    return oclSuccess("clCreateBuffer", this);
}

template <class TYPE> size_t oclBuffer::count()
{
    if (mMemPtr)
    {
        return dim(0)/sizeof(TYPE);
    }
    return 0;
}

template <class TYPE> bool oclBuffer::resize(size_t iElements, void* iHostPtr) 
{
    return resize(iElements*sizeof(TYPE), iHostPtr);
}


#endif