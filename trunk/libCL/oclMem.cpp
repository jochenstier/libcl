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
#include "oclMem.h"

//
//
//

oclMem::oclMem(oclContext& iContext, char* iName)
: oclObject(iName)
, mMemPtr(0)
, mHostPtr(0)
, mContext(iContext)
, mMapping(0)
{
}

cl_mem& oclMem::getMem ()
{
    return mMemPtr;
}

oclMem::operator cl_mem ()
{
    return mMemPtr;
}

oclMem::operator const void* ()
{
    return &mMemPtr;
}

oclMem::operator bool ()
{
    return mMemPtr != 0;
}

oclContext& oclMem::getContext()
{
    return mContext;
}

cl_map_flags oclMem::getMapping()
{
    return mMapping;
}



bool oclMem::unmap(oclDevice& iDevice)
{
    if (mMemPtr)
    {
        sStatusCL = clEnqueueUnmapMemObject(iDevice, 
                                            mMemPtr, 
                                            (void*)mHostPtr, 
                                            0, 
                                            NULL, 
                                            NULL);
        mMapping = 0;
        return oclSuccess("clEnqueueUnmapMemObject", this);
    }
    return false;
}


void oclMem::destroy()
{
    if (mMemPtr)
    {
        sStatusCL = clReleaseMemObject(mMemPtr);
        mMemPtr = 0;
        oclSuccess("clReleaseMemObject", this);
    }
}

/*
void oclMem::allocHostPtr()
{
    if (mMemFlags & CL_MEM_USE_HOST_PTR)
    {
        if (!mHostPtr)
        {
            mHostPtr = new char[byteCount()];
        }
    }
    else
    {
        mHostPtr = 0;
    }
}

void oclMem::deallocHostPtr()
{
    if (mMemFlags & CL_MEM_USE_HOST_PTR)
    {
        if (mHostPtr)
        {
            delete [] mHostPtr;
        }
    }
    else
    {
        mHostPtr = 0;
    }
}

void oclMem::resizeHostPtr(size_t iByteCount)
{
    if (mMemFlags & CL_MEM_USE_HOST_PTR)
    {
        if (byteCount() >= iByteCount)
        {
            delete [] mHostPtr;
            mHostPtr = new char[iByteCount];
        }
    }
}
*/

unsigned long oclMem::sMemoryUsed = 0;

void oclMem::incMemory(unsigned long iBytes)
{
    sMemoryUsed += iBytes;
};

void oclMem::decMemory(unsigned long iBytes)
{
    sMemoryUsed -= iBytes;
};

unsigned long oclMem::getMemory()
{
    return sMemoryUsed;
};
