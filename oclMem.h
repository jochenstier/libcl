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
#ifndef _oclMem
#define _oclMem

#include "oclContext.h"

class oclMem : public oclObject
{
    public:

        oclMem(oclContext& iContext, char* iName ="");
        oclContext& getContext();

        //
        cl_mem& getMem();
        operator cl_mem ();  
        operator const void* ();
        operator bool ();

        //
        virtual bool map(oclDevice& iDevice, cl_map_flags iMapping) = 0;
        virtual bool unmap(oclDevice& iDevice);
        cl_map_flags getMapping();
        template <class TYPE> __inline  TYPE* ptr();

        //
        virtual void destroy();

        //
        template <class RESULT> RESULT getMemObjectInfo(cl_mem_info iInfo);
        virtual size_t dim(int iAxis) = 0;

    protected:

        oclContext& mContext;
        void* mHostPtr;
        cl_map_flags mMapping;
        cl_mem mMemPtr;

    private:

        oclMem(const oclMem&);
        oclMem& operator = (const oclMem&);


    // Memory counter 
    public:

        static void incMemory(unsigned long iBytes);
        static void decMemory(unsigned long iBytes);
        static unsigned long getMemory();

    private:

        static unsigned long sMemoryUsed;

};

//
//
//

template <class TYPE> __inline  TYPE* oclMem::ptr()
{
    return (TYPE*) mHostPtr;
}

template <class RESULT> RESULT oclMem::getMemObjectInfo(cl_mem_info iInfo) 
{
    RESULT lResult;
    clGetMemObjectInfo(mMemPtr, iInfo, sizeof(RESULT), &lResult, 0);
    return lResult;
}


#endif