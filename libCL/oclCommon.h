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

    private:

        std::stringstream mStream;
        fnLogger mFunction;
        oclObject* mObject;
};

#endif