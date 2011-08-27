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
#ifndef _oclDevice
#define _oclDevice

#include "oclObject.h"

class oclContext;

class oclDevice : public oclObject
{
    public:

        oclDevice(oclContext& iContext, cl_device_id iDevice);
        ~oclDevice();
        operator cl_device_id();
        operator bool ();

        //
        template <class RESULT> RESULT getContextInfo(cl_image_info iInfo);

        //
        operator cl_command_queue();
        cl_command_queue getQueue();

    private:

        oclContext& mContext;

        cl_device_id mDevice;
        cl_command_queue mCommandQueue;

    private:

        oclDevice(const oclDevice&);
        oclDevice& operator = (const oclDevice&);
};

//
//
//

template <class RESULT> RESULT oclDevice::getContextInfo(cl_image_info iInfo) 
{
    RESULT lResult;
    clGetContextInfo(*this, iInfo, sizeof(RESULT), &lResult, 0);
    return lResult;
}


#endif
