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
#include "oclBuffer.h"

oclBuffer::oclBuffer(oclContext& iContext, char* iName, int iType)
: oclMem(iContext, iName)
, mType(iType)
{
};

//
//
//

bool oclBuffer::map(cl_map_flags iMapping, int iDevice)
{
    if (mMemPtr)
    {
        mHostPtr = (unsigned char*)clEnqueueMapBuffer(mContext.getDevice(iDevice), 
                                                       mMemPtr, 
                                                       CL_TRUE, 
                                                       iMapping, 
                                                       0, 
                                                       getMemObjectInfo<size_t>(CL_MEM_SIZE), 
                                                       0, 
                                                       NULL, 
                                                       NULL, 
                                                       &sStatusCL);
        mMapping = iMapping;
        return oclSuccess("clEnqueueMapBuffer", this);
    }
    Log(ERR, this) << "Invalid cl_mem";
    return false;
}

bool oclBuffer::write(int iDevice)
{
    if (mMemPtr)
    {
        sStatusCL = clEnqueueWriteBuffer(mContext.getDevice(iDevice),
                                         mMemPtr,
                                         CL_TRUE,
                                         0,
                                         getMemObjectInfo<size_t>(CL_MEM_SIZE),
                                         mHostPtr,
                                         0,
                                         0,
                                         0);
        return oclSuccess("clEnqueueWriteBuffer", this);
    }
    Log(ERR, this) << "Invalid cl_mem";
    return false;
}

bool oclBuffer::read(int iDevice)
{
    if (mMemPtr)
    {
        sStatusCL = clEnqueueReadBuffer(mContext.getDevice(iDevice),
                                         mMemPtr,
                                         CL_TRUE,
                                         0,
                                         getMemObjectInfo<size_t>(CL_MEM_SIZE),
                                         mHostPtr,
                                         0,
                                         0,
                                         0);
        return oclSuccess("clEnqueueReadBuffer", this);
    }
    Log(ERR, this) << "Invalid cl_mem";
    return false;
}

bool oclBuffer::resize(size_t iSize, void* iHostPtr)
{
    if (mMemPtr)
    {
        cl_mem_flags lMemFlags = getMemObjectInfo<cl_mem_flags>(CL_MEM_FLAGS); 
        destroy();
        mMemPtr = clCreateBuffer(mContext, lMemFlags, iSize, iHostPtr, &sStatusCL);
        return oclSuccess("clCreateBuffer", this);
    }
    else 
    {
        Log(ERR, this) << "Invalid cl_mem";
    }
    return false;
}

//
//
//

size_t oclBuffer::dim(int iAxis)
{
    switch (iAxis)
    {
        case 0: return getMemObjectInfo<size_t>(CL_MEM_SIZE);
    }
    return 0;
};

//
//
//

int oclBuffer::getType()
{
    return mType;
};
