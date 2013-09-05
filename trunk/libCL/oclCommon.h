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
#ifndef _oclCommon
#define _oclCommon

#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>

using namespace std;

#pragma warning(disable: 4244) // possible loss of data
#pragma warning(disable: 4355) // this used in initilizer list

#include "oclObject.h"

void oclInit(char* iRootPath);

#define UU_LIBCL 1

//
// Error handling
//
extern cl_int sStatusCL;
const char* oclError();
bool oclSuccess(char* iFunction, oclObject* iObject = 0);

typedef void (*fnLogger) (const char* iMessage, oclObject* iObject);
extern fnLogger ERR;
extern fnLogger WARN;
extern fnLogger INFO;
extern fnLogger KERNEL;

class Log
{
    public:

        Log(fnLogger iFunction, oclObject* iObject=0, const char* iFile=0, int iLine=0);
        ~Log();

        Log& operator<< (int iValue);
        Log& operator<< (unsigned int iValue);
        Log& operator<< (cl_long iValue);
        Log& operator<< (cl_ulong iValue);
        Log& operator<< (float iValue);
        Log& operator<< (char* iValue);

        Log& operator<< (const char* iValue);
        Log& operator<< (void* iValue);
        Log& operator<< (char iValue);
        Log& operator<< (unsigned char iValue);
        Log& operator<< (double iValue);

        Log& operator<< (cl_float4& iValue);
        Log& operator<< (cl_float2& iValue);

    private:

        std::stringstream mStream;
        fnLogger mFunction;
        oclObject* mObject;
};



//
// Printers
// 

template <class TYPE> void prtEntryN(char* iBuffer, TYPE& iBlock) 
{
    strcat(iBuffer, " prtEntryN not implemented for given type ");
}
template <> void prtEntryN<cl_uchar2> (char* iBuffer, cl_uchar2& iBlock);
template <> void prtEntryN<cl_char2> (char* iBuffer, cl_char2& iBlock);
template <> void prtEntryN<cl_short2> (char* iBuffer, cl_short2& iBlock);
template <> void prtEntryN<cl_ushort2> (char* iBuffer, cl_ushort2& iBlock);
template <> void prtEntryN<cl_int2> (char* iBuffer, cl_int2& iBlock);
template <> void prtEntryN<cl_uint2> (char* iBuffer, cl_uint2& iBlock);
template <> void prtEntryN<cl_float2> (char* iBuffer, cl_float2& iBlock);
template <> void prtEntryN<cl_double2> (char* iBuffer, cl_double2& iBlock);

template <> void prtEntryN<cl_uchar4> (char* iBuffer, cl_uchar4& iBlock);
template <> void prtEntryN<cl_char4> (char* iBuffer, cl_char4& iBlock);
template <> void prtEntryN<cl_short4> (char* iBuffer, cl_short4& iBlock);
template <> void prtEntryN<cl_ushort4> (char* iBuffer, cl_ushort4& iBlock);
template <> void prtEntryN<cl_int4> (char* iBuffer, cl_int4& iBlock);
template <> void prtEntryN<cl_uint4> (char* iBuffer, cl_uint4& iBlock);
template <> void prtEntryN<cl_float4> (char* iBuffer, cl_float4& iBlock);
template <> void prtEntryN<cl_double4> (char* iBuffer, cl_double4& iBlock);

template <class TYPE> void prtEntry1(char* iBuffer, TYPE& iBlock) 
{
    strcat(iBuffer, " prtEntry1 not implemented for given type ");
}
template <> void prtEntry1<cl_char> (char* iBuffer, cl_char& iValue);
template <> void prtEntry1<cl_uchar> (char* iBuffer, cl_uchar& iValue);
template <> void prtEntry1<cl_short> (char* iBuffer, cl_short& iValue);
template <> void prtEntry1<cl_ushort> (char* iBuffer, cl_ushort& iValue);
template <> void prtEntry1<cl_int> (char* iBuffer, cl_int& iValue);
template <> void prtEntry1<cl_uint> (char* iBuffer, cl_uint& iValue);
template <> void prtEntry1<cl_float> (char* iBuffer, cl_float& iValue);
template <> void prtEntry1<cl_double> (char* iBuffer, cl_double& iValue);


#endif