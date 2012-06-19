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

		static const int _char = 0;
		static const int _uchar = 1;
		static const int _short = 2;
		static const int _ushort = 3;
		static const int _int = 4;
		static const int _uint = 5;
		static const int _long = 6;
		static const int _ulong = 7;
		static const int _half = 8;
		static const int _float = 9;
		static const int _double = 10;

		static const int _char4 = 11;
		static const int _uchar4 = 12;
		static const int _short4 = 13;
		static const int _ushort4 = 14;
		static const int _int4 = 15;
		static const int _uint4 = 16;
		static const int _long4 = 17;
		static const int _ulong4 = 18;
		static const int _float4 = 19;
		static const int _double4 = 20;

		static const int _char2 = 21;
		static const int _uchar2 = 22;
		static const int _short2 = 23;
		static const int _ushort2 = 24;
		static const int _int2 = 25;
		static const int _uint2 = 26;
		static const int _long2 = 27;
		static const int _ulong2 = 28;
		static const int _float2 = 29;
		static const int _double2 = 30;

        oclBuffer(oclContext& iContext, char* iName="oclBuffer", int iType = _char);

        //
        template <class TYPE> bool create(cl_mem_flags iMemFlags, size_t iElements, void* iHostPtr=0);
        virtual bool map(cl_map_flags iMapping, int iDevice = 0);
        virtual bool read(int iDevice = 0);
        virtual bool write(int iDevice = 0);

        //
        template <class TYPE> bool resize(size_t iElements, void* iHostPtr=0) ;
        virtual bool resize(size_t iSize, void* iHostPtr = 0);

        //
        template <class TYPE> size_t count();
        virtual size_t dim(int iAxis);

		int getType();

    private:

		int mType;

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