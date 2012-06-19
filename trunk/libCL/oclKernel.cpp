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
#include <math.h>

#include "oclKernel.h"
#include "oclDevice.h"
#include "oclCommon.h"

oclKernel::oclKernel(oclProgram& iProgram)
: oclObject()
, mKernel()
, mEvent(0)
, mProgram(iProgram)
, mProfiling(false)
, mStartTime(0)
, mEndTime(0)
{
}

oclKernel::oclKernel(oclProgram& iProgram, cl_kernel& iKernel)
: oclObject()
, mKernel(iKernel)
, mEvent(0)
, mProgram(iProgram)
, mStartTime(0)
, mEndTime(0)
{
    if (mKernel)
    {
        size_t lSize = 0;
        clGetKernelInfo(mKernel,CL_KERNEL_FUNCTION_NAME,0,0,&lSize);
        oclSuccess("clGetKernelInfo", this);
        mName = new char[lSize];
        clGetKernelInfo(mKernel,CL_KERNEL_FUNCTION_NAME,lSize,mName,0);
        oclSuccess("clGetKernelInfo", this);
        sStatusCL = clRetainKernel(mKernel);
        oclSuccess("clRetainKernel", this);
    }
}

oclKernel::~oclKernel()
{
    if (mKernel)
    {
        sStatusCL = clReleaseKernel(mKernel);
        oclSuccess("clReleaseKernel", this);
        delete [] mName;
        mName = 0;
        cl_uint lCount;
        clGetEventInfo (mEvent, CL_EVENT_REFERENCE_COUNT, sizeof(cl_uint), &lCount, NULL);
        if (mEvent)
        {
            clReleaseEvent(mEvent);
            mEvent = 0;
        }
    }
}

oclKernel::operator cl_kernel ()
{ 
    return mKernel; 
}

cl_kernel& oclKernel::getKernel()
{ 
    return mKernel; 
}

cl_event* oclKernel::getEvent()
{ 
    if (mProfiling)
    {
        if (mEvent)
        {
            sStatusCL = clReleaseEvent(mEvent); //this causes a memory leak but at least profing works on NVIDIA
            oclSuccess("clReleaseEvent", this);
            mEvent = 0;
        }
        return &mEvent; 
    }
    else
    {
        mEndTime = 0;
        mStartTime = 0;
        return NULL;
    }
}

oclProgram& oclKernel::getProgram()
{ 
    return mProgram; 
}

//
//
//

void oclKernel::localSize2D(oclDevice& iDevice, size_t lGlobalSize[2], size_t lLocalSize[2], int iW, int iH)
{
    size_t lSize = getKernelWorkGroupInfo<size_t>(CL_KERNEL_WORK_GROUP_SIZE, iDevice);
    lLocalSize[0] = floor(sqrt(1.0*lSize));
    lLocalSize[1] = floor(sqrt(1.0*lSize));
    lGlobalSize[0] = ceil((float)iW/lLocalSize[0])*lLocalSize[0];
    lGlobalSize[1] = ceil((float)iH/lLocalSize[1])*lLocalSize[1];
}


//
//
//

oclKernel& oclKernel::operator = (cl_kernel iKernel)
{
    if (mKernel)
    {
        sStatusCL = clReleaseKernel(mKernel);
        oclSuccess("clReleaseKernel", this);
        delete [] mName;
        mName = 0;
        cl_uint lCount;
        clGetEventInfo (mEvent, CL_EVENT_REFERENCE_COUNT, sizeof(cl_uint), &lCount, NULL);
        if (lCount)
        {
            clReleaseEvent(mEvent);
            mEvent = 0;
        }
    }
    mKernel = iKernel;
    if (iKernel)
    {
        size_t lSize = 0;
        clGetKernelInfo(mKernel,CL_KERNEL_FUNCTION_NAME,0,0,&lSize);
        oclSuccess("clGetKernelInfo", this);
        mName = new char[lSize];
        clGetKernelInfo(mKernel,CL_KERNEL_FUNCTION_NAME,lSize,mName,0);
        oclSuccess("clGetKernelInfo", this);
        sStatusCL = clRetainKernel(mKernel);
        oclSuccess("clRetainKernel", this);
    }
    return *this;
}

//
//
//

void oclKernel::profile(bool iState)
{
    mProfiling = iState;
}

cl_ulong oclKernel::getStartTime()
{
    cl_ulong lStartTime = 0;
    if (mEvent)
    {
        clWaitForEvents (1, &mEvent);
        sStatusCL = clGetEventProfilingInfo(mEvent,
                                          CL_PROFILING_COMMAND_START,
                                          sizeof(cl_ulong),
                                          &lStartTime,
                                          0);
        if (oclSuccess("clGetEventProfilingInfo", this))
        {
            mStartTime = lStartTime;
        }
    }
    return mStartTime;
}
cl_ulong oclKernel::getEndTime()
{
    cl_ulong lEndtime = 0;
    if (mEvent)
    {
        clWaitForEvents (1, &mEvent);
        sStatusCL = clGetEventProfilingInfo(mEvent,
                                          CL_PROFILING_COMMAND_END,
                                          sizeof(cl_ulong),
                                          &lEndtime,
                                          0);
        if (oclSuccess("clGetEventProfilingInfo", this))
        {
            mEndTime = lEndtime;
        }
    }
    return mEndTime;
}

//
//
//
