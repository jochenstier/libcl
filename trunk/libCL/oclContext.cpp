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
#include <windows.h>

#include "oclContext.h" 
#include "oclDevice.h" 

char* oclContext::VENDOR_NVIDIA = "NVIDIA Corporation";
char* oclContext::VENDOR_AMD = "Advanced Micro Devices, Inc.";
char* oclContext::VENDOR_INTEL = "Intel Corporation";
char* oclContext::VENDOR_UNKNOWN = "Unknown Vendor";

oclContext::oclContext(cl_context iContext, char* iVendor)
: oclObject("")
, mContext(iContext)
, mVendor(VENDOR_UNKNOWN)
{
    if (!strcmp(iVendor, VENDOR_NVIDIA))
    {
        mVendor = VENDOR_NVIDIA;
    }
    else if (!strcmp(iVendor, VENDOR_AMD))
    {
        mVendor = VENDOR_AMD;
    }
    else if (!strncmp(iVendor, VENDOR_INTEL, 5))
    {
        mVendor = VENDOR_INTEL;
    }

    setName(mVendor);

    size_t lDeviceCount;
    sStatusCL = clGetContextInfo(mContext, CL_CONTEXT_DEVICES, 0, NULL, &lDeviceCount);
    oclSuccess("clGetContextInfo", this);

    cl_device_id* lDevice = new cl_device_id[lDeviceCount];
    clGetContextInfo(mContext, CL_CONTEXT_DEVICES, lDeviceCount, lDevice, NULL);
    for (cl_uint i=0; i<lDeviceCount/sizeof(cl_device_id); i++)
    {
        mDevices.push_back(new oclDevice(*this, lDevice[i]));
    }
    delete [] lDevice;
};

oclContext::operator cl_context()  
{  
    return mContext;  
}

vector<oclDevice*>& oclContext::getDevices()
{
    return mDevices;
};

oclDevice& oclContext::getDevice(int iIndex)
{
    return *mDevices[iIndex];
};

typedef CL_API_ENTRY cl_int (CL_API_CALL *clGetGLContextInfoKHR_fn)(const cl_context_properties * /* properties */,
                                                                    cl_gl_context_info /* param_name */,
                                                                    size_t /* param_value_size */,
                                                                    void * /* param_value */,
                                                                    size_t * /*param_value_size_ret*/);


oclContext* oclContext::create(const char* iVendor, int iDeviceType)
{
    cl_uint lPlatformCount = 0;
    sStatusCL = clGetPlatformIDs(0, NULL, &lPlatformCount);
    oclSuccess("clGetPlatformIDs");

    clGetGLContextInfoKHR_fn _clGetGLContextInfoKHR = (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddress("clGetGLContextInfoKHR");

    cl_platform_id lPlatform[100];
    sStatusCL = clGetPlatformIDs(lPlatformCount, lPlatform, NULL);
    oclSuccess("clGetPlatformIDs");

    char lBuffer[200];
    for (cl_uint i=0; i < lPlatformCount; i++) 
    {
        sStatusCL = clGetPlatformInfo(lPlatform[i],
                                       CL_PLATFORM_VENDOR,
                                       sizeof(lBuffer),
                                       lBuffer,
                                       NULL);
        oclSuccess("clGetPlatformInfo");

        cl_context_properties GL_PROPS[] = 
        {
            CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
            CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
            CL_CONTEXT_PLATFORM, (cl_context_properties)lPlatform[i], 
            0
        };

        cl_context_properties CL_PROPS[] = 
        {
            CL_CONTEXT_PLATFORM, (cl_context_properties)lPlatform[i], 
            0
        };

        if (!strncmp(lBuffer, iVendor, 5))  // compare only first 5 letters -- Intel starts with "Intel" and "Intel(R)"
        {
            switch (iDeviceType)
            {
            case CL_DEVICE_TYPE_GPU:
                // gpu context
                cl_device_id lDevices[100];
                cl_uint lDeviceCount;
                sStatusCL = clGetDeviceIDs(lPlatform[i], CL_DEVICE_TYPE_GPU, 100, lDevices, &lDeviceCount);
                if (!oclSuccess("clGetDeviceIDs"))
                {
                    continue;
                }

                if (lDeviceCount)
                {
                    size_t lDeviceGLCount = 0;
                    cl_device_id lDeviceGL; 
                    sStatusCL = _clGetGLContextInfoKHR(GL_PROPS, CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, sizeof(cl_device_id), &lDeviceGL, &lDeviceGLCount);
                    if (!oclSuccess("clGetDeviceIDs"))
                    {
                        //continue; AMD drivers produce an error here if not OpenGL context is present
                    }

                    if (lDeviceGLCount)
                    {
                        // gpu context with sharing enabled
                        cl_context lContextCL = clCreateContext(GL_PROPS, lDeviceCount, lDevices, NULL, NULL, &sStatusCL);
                        if (!oclSuccess("clCreateContext"))
                        {
                            continue;
                        }
                        return new oclContext(lContextCL, lBuffer);
                    }
                    else
                    {
                        // gpu context without sharing
                        cl_context lContextCL = clCreateContext(CL_PROPS, lDeviceCount, lDevices, NULL, NULL, &sStatusCL);
                        if (!oclSuccess("clCreateContext"))
                        {
                            continue;
                        }
                        return new oclContext(lContextCL, lBuffer);
                    }
                }
                break;

            case CL_DEVICE_TYPE_CPU:
                // cpu context
                cl_context lContextCL = clCreateContextFromType(CL_PROPS, CL_DEVICE_TYPE_CPU, NULL, NULL, &sStatusCL);
                if (!oclSuccess("clCreateContextFromType"))
                {
                    continue;
                }
                return new oclContext(lContextCL, lBuffer);
                break;
            }
        }
    }
    return 0;
}


//
//
//
char* oclContext::getVendor()
{
    return mVendor;
};
