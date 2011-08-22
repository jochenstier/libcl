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
#ifndef _oclContext
#define _oclContext

#include <vector> 

#include "oclObject.h"
#include "oclDevice.h"
#include "oclCommon.h"

class oclContext : public oclObject
{
    public:

        static char* VENDOR_NVIDIA;
        static char* VENDOR_AMD;
        static char* VENDOR_INTEL;
        static oclContext* create(const char* iVendor, int iDeviceType);

        oclContext(cl_context iContext, char* iVendor=0);
        operator cl_context();

        //
        template <class RESULT> RESULT getContextInfo(cl_image_info iInfo);

        oclDevice& getDevice(int iIndex);
        vector<oclDevice*>& getDevices();

    private:

        cl_context mContext;
        vector<oclDevice*> mDevices;

    private:

        char* getVendor(char* iName);

    private:

        oclContext(const oclContext&);
        oclContext& operator = (const oclContext&);
};


//
//
//

template <class RESULT> RESULT oclContext::getContextInfo(cl_image_info iInfo) 
{
    RESULT lResult;
    clGetContextInfo(*this, iInfo, sizeof(RESULT), &lResult, 0);
    return lResult;
}


#endif
