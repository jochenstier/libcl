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
#ifndef _oclKernel
#define _oclKernel

#include "oclObject.h"

class oclProgram;

class oclKernel : public oclObject
{
    public: 

        oclKernel(oclProgram& iProgram);
        oclKernel(oclProgram& iProgram, cl_kernel& iKernel);
       ~oclKernel();

        oclKernel& operator = (cl_kernel iKernel);
        operator cl_kernel ();

        //
        template <class RETURN> RETURN getKernelWorkGroupInfo(cl_uint iValue, cl_device_id iDevice);
        template <class RETURN> RETURN getKernelInfo(cl_kernel_info iInfo);

        //
        void profile(cl_ulong& iStartTime, cl_ulong& iEndTime);

        //
        cl_kernel& getKernel();
        cl_event* getEvent();
        oclProgram& getProgram();

    private:

        oclProgram& mProgram;

        cl_kernel mKernel;
        cl_event mEvent;

    private:

        oclKernel(const oclKernel&);
        oclKernel& operator = (const oclKernel&);
};      

//
//
//

template <class RETURN> RETURN oclKernel::getKernelWorkGroupInfo(cl_uint iValue, cl_device_id iDevice)
{
    static RETURN lResult;
    clGetKernelWorkGroupInfo(mKernel, iDevice, iValue, sizeof(RETURN), &lResult, NULL);
    return lResult;
}

template <class RETURN> RETURN oclKernel::getKernelInfo(cl_kernel_info iInfo)
{
    static RETURN lResult;
    clGetKernelInfo(mKernel, iValue, sizeof(RETURN), &lResult, NULL);
    return lResult;
}

//
//
//

#define KERNEL_VALIDATE(kernel)\
if (!kernel)\
{\
    Log(ERR, this) << "Kernel " << #kernel << " not found in " << mName;\
    return 0;\
}\


#endif