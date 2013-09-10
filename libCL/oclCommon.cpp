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
#include "oclCommon.h"
#include "oclProgram.h"

#include <iostream>

cl_int sStatusCL = CL_SUCCESS;

#define CL_ERRORS 64

const char* clErrors[CL_ERRORS];

const char* oclError()
{
    if (-sStatusCL < CL_ERRORS)
    {
        return clErrors[-sStatusCL]; 
    }
    else
    {
        static char sError[400];
        sprintf(sError, "Unknown OpenCL error: %d", sStatusCL);
        return sError;
    }
}

bool oclSuccess(char* iFunction, oclObject* iObject)
{
    if (sStatusCL != CL_SUCCESS)
    {
        Log(ERR, iObject) << "Failure in call to " << iFunction << " : " << oclError(); 
    }
    return sStatusCL == CL_SUCCESS;
};

void info(const char* iValue, oclObject* iObject)
{
    if (iObject)
    {
        cout << iObject->getName() << ":" << iValue << "\n";
    }
    else 
    {
        cout << iValue << "\n";
    }
}
void warn(const char* iValue, oclObject* iObject)
{
    if (iObject)
    {
        cout << "(warning):" << iObject->getName() << ":" << iValue << "\n";
    }
    else 
    {
        cout << "(warning):" << iValue << "\n";
    }
}
void error(const char* iValue, oclObject* iObject)
{
    if (iObject)
    {
        cout << "(error):" << iObject->getName() << ":" << iValue << "\n";
    }
    else 
    {
        cout << "(error):" << iValue << "\n";
    }
}

fnLogger ERR = error;
fnLogger WARN = warn;
fnLogger INFO = info;
fnLogger KERNEL = error;

void oclInit(char* iRootPath)
{
    oclProgram::setRootPath(iRootPath);

    clErrors[-CL_SUCCESS] = "CL_SUCCESS";
    clErrors[-CL_DEVICE_NOT_FOUND] = "CL_DEVICE_NOT_FOUND";
    clErrors[-CL_DEVICE_NOT_AVAILABLE] = "CL_DEVICE_NOT_AVAILABLE";
    clErrors[-CL_COMPILER_NOT_AVAILABLE] = "CL_COMPILER_NOT_AVAILABLE";
    clErrors[-CL_MEM_OBJECT_ALLOCATION_FAILURE] = "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    clErrors[-CL_OUT_OF_RESOURCES] = "CL_OUT_OF_RESOURCES";
    clErrors[-CL_OUT_OF_HOST_MEMORY] = "CL_OUT_OF_HOST_MEMORY";
    clErrors[-CL_PROFILING_INFO_NOT_AVAILABLE] = "CL_PROFILING_INFO_NOT_AVAILABLE";
    clErrors[-CL_MEM_COPY_OVERLAP] = "CL_MEM_COPY_OVERLAP";
    clErrors[-CL_IMAGE_FORMAT_MISMATCH] = "CL_IMAGE_FORMAT_MISMATCH";
    clErrors[-CL_IMAGE_FORMAT_NOT_SUPPORTED] = "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    clErrors[-CL_BUILD_PROGRAM_FAILURE] = "CL_BUILD_PROGRAM_FAILURE";
    clErrors[-CL_MAP_FAILURE] = "CL_MAP_FAILURE";
    clErrors[-CL_INVALID_VALUE] = "CL_INVALID_VALUE";
    clErrors[-CL_INVALID_DEVICE_TYPE] = "CL_INVALID_DEVICE_TYPE";
    clErrors[-CL_INVALID_PLATFORM] = "CL_INVALID_PLATFORM";
    clErrors[-CL_INVALID_DEVICE] = "CL_INVALID_DEVICE";
    clErrors[-CL_INVALID_CONTEXT] = "CL_INVALID_CONTEXT";
    clErrors[-CL_INVALID_QUEUE_PROPERTIES] = "CL_INVALID_QUEUE_PROPERTIES";
    clErrors[-CL_INVALID_COMMAND_QUEUE] = "CL_INVALID_COMMAND_QUEUE";
    clErrors[-CL_INVALID_HOST_PTR] = "CL_INVALID_HOST_PTR";
    clErrors[-CL_INVALID_MEM_OBJECT] = "CL_INVALID_MEM_OBJECT";
    clErrors[-CL_INVALID_IMAGE_FORMAT_DESCRIPTOR] = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    clErrors[-CL_INVALID_IMAGE_SIZE] = "CL_INVALID_IMAGE_SIZE";
    clErrors[-CL_INVALID_SAMPLER] = "CL_INVALID_SAMPLER";
    clErrors[-CL_INVALID_BINARY] = "CL_INVALID_BINARY";
    clErrors[-CL_INVALID_BUILD_OPTIONS] = "CL_INVALID_BUILD_OPTIONS";
    clErrors[-CL_INVALID_PROGRAM] = "CL_INVALID_PROGRAM";
    clErrors[-CL_INVALID_PROGRAM_EXECUTABLE] = "CL_INVALID_PROGRAM_EXECUTABLE";
    clErrors[-CL_INVALID_KERNEL_NAME] = "CL_INVALID_KERNEL_NAME";
    clErrors[-CL_INVALID_KERNEL_DEFINITION] = "CL_INVALID_KERNEL_DEFINITION";
    clErrors[-CL_INVALID_KERNEL] = "CL_INVALID_KERNEL";
    clErrors[-CL_INVALID_ARG_INDEX] = "CL_INVALID_ARG_INDEX";
    clErrors[-CL_INVALID_ARG_VALUE] = "CL_INVALID_ARG_VALUE";
    clErrors[-CL_INVALID_ARG_SIZE] = "CL_INVALID_ARG_SIZE";
    clErrors[-CL_INVALID_KERNEL_ARGS] = "CL_INVALID_KERNEL_ARGS";
    clErrors[-CL_INVALID_WORK_DIMENSION] = "CL_INVALID_WORK_DIMENSION";
    clErrors[-CL_INVALID_WORK_GROUP_SIZE] = "CL_INVALID_WORK_GROUP_SIZE";
    clErrors[-CL_INVALID_WORK_ITEM_SIZE] = "CL_INVALID_WORK_ITEM_SIZE";
    clErrors[-CL_INVALID_GLOBAL_OFFSET] = "CL_INVALID_GLOBAL_OFFSET";
    clErrors[-CL_INVALID_EVENT_WAIT_LIST] = "CL_INVALID_EVENT_WAIT_LIST";
    clErrors[-CL_INVALID_EVENT] = "CL_INVALID_EVENT";
    clErrors[-CL_INVALID_OPERATION] = "CL_INVALID_OPERATION";
    clErrors[-CL_INVALID_GL_OBJECT] = "CL_INVALID_GL_OBJECT";
    clErrors[-CL_INVALID_BUFFER_SIZE] = "CL_INVALID_BUFFER_SIZE";
    clErrors[-CL_INVALID_MIP_LEVEL] = "CL_INVALID_MIP_LEVEL";
    clErrors[-CL_INVALID_GLOBAL_WORK_SIZE] = "CL_INVALID_GLOBAL_WORK_SIZE";
}



Log::Log(fnLogger iFunction, oclObject* iObject, const char* iFile, int iLine)
: mStream()
, mFunction(iFunction)
, mObject(iObject)
{
    if (iFile)
    {
        mStream << "(" << iFile << ":" << iLine << "):";
    }
    if (iFunction == ERR)
    {
        if (iObject)
        {
            iObject->setError();
        }
    }
}

Log::~Log()
{
    mStream << "\0";
    if (mFunction)
    {
        (*mFunction)(mStream.str().c_str(), mObject);
    }
}

Log& Log::operator<< (int iValue) 
{
    mStream << iValue;
    return *this;
}

Log& Log::operator<< (unsigned int iValue) 
{
    mStream << iValue;
    return *this;
}

Log& Log::operator<< (cl_long iValue) 
{
    mStream << iValue;
    return *this;
}

Log& Log::operator<< (cl_ulong iValue) 
{
    mStream << iValue;
    return *this;
}

Log& Log::operator<< (float iValue) 
{
    mStream << iValue;
    return *this;
}

Log& Log::operator<< (char* iValue) 
{
    mStream << iValue;
    return *this;
}

Log& Log::operator<< (const char* iValue) 
{
    mStream << iValue;
    return *this;
}

Log& Log::operator<< (void* iValue) 
{
    mStream << iValue;
    return *this;
}

Log& Log::operator<< (char iValue) 
{
    mStream << iValue;
    return *this;
}

Log& Log::operator<< (unsigned char iValue)
{
    mStream << iValue;
    return *this;
}

Log& Log::operator<< (double iValue) 
{
    mStream << iValue;
    return *this;
}

Log& Log::operator<< (cl_float2& iValue) 
{
    mStream << "( " << iValue.s[0] << "," << iValue.s[1] << ")";
    return *this;
}
Log& Log::operator<< (cl_float4& iValue) 
{
    mStream << "( " << iValue.s[0] << "," << iValue.s[1] << "," << iValue.s[2] << "," << iValue.s[3] << ")";
    return *this;
}




//
// printers
//
template <> void prtEntry<cl_char2> (char* iBuffer, cl_char2& iBlock) 
{
    char lBuffer[100];
    sprintf(lBuffer, "%hd, %hd", (short)iBlock.s[0], (short)iBlock.s[1]);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_uchar2> (char* iBuffer, cl_uchar2& iBlock) 
{
    char lBuffer[100];
    sprintf(lBuffer, "%hu, %hu", (unsigned short)iBlock.s[0], (unsigned short)iBlock.s[1]);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_short2> (char* iBuffer, cl_short2& iBlock) 
{
    char lBuffer[100];
    sprintf(lBuffer, "%hd, %hd", iBlock.s[0], iBlock.s[1]);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_ushort2> (char* iBuffer, cl_ushort2& iBlock) 
{
    char lBuffer[100];
    sprintf(lBuffer, "%hu, %hu", iBlock.s[0], iBlock.s[1]);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_int2> (char* iBuffer, cl_int2& iBlock) 
{
    char lBuffer[100];
    sprintf(lBuffer, "%d, %d", iBlock.s[0], iBlock.s[1]);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_uint2> (char* iBuffer, cl_uint2& iBlock) 
{
    char lBuffer[100];
    sprintf(lBuffer, "%u, %u", iBlock.s[0], iBlock.s[1]);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_float2> (char* iBuffer, cl_float2& iBlock) 
{
    char lBuffer[100];
    sprintf(lBuffer, "%f, %f", iBlock.s[0], iBlock.s[1]);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_double2> (char* iBuffer, cl_double2& iBlock) 
{
    char lBuffer[100];
    sprintf(lBuffer, "%lf, %lf", iBlock.s[0], iBlock.s[1]);
    strcat(iBuffer, lBuffer);
}



template <> void prtEntry<cl_char4> (char* iBuffer, cl_char4& iBlock) 
{
    char lBuffer[100];
    sprintf(lBuffer, "%hd, %hd, %hd, %hd", (short)iBlock.s[0], (short)iBlock.s[1], (short)iBlock.s[2], (short)iBlock.s[3]);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_uchar4> (char* iBuffer, cl_uchar4& iBlock) 
{
    char lBuffer[100];
    sprintf(lBuffer, "%hu, %hu, %hu, %hu", (unsigned short)iBlock.s[0], (unsigned short)iBlock.s[1], (unsigned short)iBlock.s[2], (unsigned short)iBlock.s[3]);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_short4> (char* iBuffer, cl_short4& iBlock) 
{
    char lBuffer[100];
    sprintf(lBuffer, "%hd, %hd, %hd, %hd", iBlock.s[0], iBlock.s[1], iBlock.s[2], iBlock.s[3]);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_ushort4> (char* iBuffer, cl_ushort4& iBlock) 
{
    char lBuffer[100];
    sprintf(lBuffer, "%hu, %hu, %hu, %hu", iBlock.s[0], iBlock.s[1], iBlock.s[2], iBlock.s[3]);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_int4> (char* iBuffer, cl_int4& iBlock) 
{
    char lBuffer[100];
    sprintf(lBuffer, "%d, %d, %d, %d", iBlock.s[0], iBlock.s[1], iBlock.s[2], iBlock.s[3]);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_uint4> (char* iBuffer, cl_uint4& iBlock) 
{
    char lBuffer[100];
    sprintf(lBuffer, "%u, %u, %u, %u", iBlock.s[0], iBlock.s[1], iBlock.s[2], iBlock.s[3]);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_float4> (char* iBuffer, cl_float4& iBlock) 
{
    char lBuffer[100];
    sprintf(lBuffer, "%f, %f, %f, %f", iBlock.s[0], iBlock.s[1], iBlock.s[2], iBlock.s[3]);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_double4> (char* iBuffer, cl_double4& iBlock) 
{
    char lBuffer[100];
    sprintf(lBuffer, "%lf, %lf, %lf, %lf", iBlock.s[0], iBlock.s[1], iBlock.s[2], iBlock.s[3]);
    strcat(iBuffer, lBuffer);
}




template <> void prtEntry<cl_char> (char* iBuffer, cl_char& iValue) 
{
    char lBuffer[100];
    int lValue = iValue;
    sprintf(lBuffer, "%d", (int)lValue);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_uchar> (char* iBuffer, cl_uchar& iValue) 
{
    char lBuffer[100];
    int lValue = iValue;
    sprintf(lBuffer, "%u", (unsigned int)lValue);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_short> (char* iBuffer, cl_short& iValue) 
{
    char lBuffer[100];
    int lValue = iValue;
    sprintf(lBuffer, "%hd", lValue);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_ushort> (char* iBuffer, cl_ushort& iValue) 
{
    char lBuffer[100];
    int lValue = iValue;
    sprintf(lBuffer, "%hu", lValue);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_int> (char* iBuffer, cl_int& iValue) 
{
    char lBuffer[100]; 
    int lValue = iValue;
    sprintf(lBuffer, "%d", lValue);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_uint> (char* iBuffer, cl_uint& iValue) 
{
    char lBuffer[100];
    int lValue = iValue;
    sprintf(lBuffer, "%u", lValue);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_float> (char* iBuffer, cl_float& iValue) 
{
    char lBuffer[100];
    sprintf(lBuffer, "%f", iValue);
    strcat(iBuffer, lBuffer);
}
template <> void prtEntry<cl_double> (char* iBuffer, cl_double& iValue) 
{
    char lBuffer[100];
    sprintf(lBuffer, "%lf", iValue);
    strcat(iBuffer, lBuffer);
}
