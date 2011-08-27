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
#include "oclKernel.h"
#include "oclCommon.h"

oclKernel::oclKernel(oclProgram& iProgram)
: oclObject()
, mKernel()
, mEvent(0)
, mProgram(iProgram)
{
}

oclKernel::oclKernel(oclProgram& iProgram, cl_kernel& iKernel)
: oclObject()
, mKernel(iKernel)
, mEvent(0)
, mProgram(iProgram)
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
    return &mEvent; 
}

oclProgram& oclKernel::getProgram()
{ 
    return mProgram; 
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

void oclKernel::profile(cl_ulong& iStartTime, cl_ulong& iEndTime)
{
    if (mEvent)
    {
        clWaitForEvents (1, &mEvent);

        size_t lRead;
        clGetEventProfilingInfo(mEvent,
                                CL_PROFILING_COMMAND_START,
                                sizeof(cl_ulong),
                                &iStartTime,
                                &lRead);
        clGetEventProfilingInfo(mEvent,
                                CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong),
                                &iEndTime,
                                &lRead);
    }
}

//
//
//
