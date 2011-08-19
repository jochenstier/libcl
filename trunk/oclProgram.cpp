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
#include "oclProgram.h"

#include <stdio.h>
#include <direct.h>

oclProgram::oclProgram(oclContext& iContext, char* iName)
: oclObject(iName)
, mProgram(0)
, mContext(iContext)
{
}

oclProgram::~oclProgram()
{
    if (mProgram)
    {
        sStatusCL = clReleaseProgram(mProgram);
        oclSuccess("clReleaseProgram", this);
    }

    clrSource();
}

//
//
//

oclContext& oclProgram::getContext()
{
    return mContext;
};

oclProgram::operator cl_program ()
{ 
    return mProgram; 
}

int oclProgram::compile()
{
    if (mProgram)
    {
        sStatusCL = clReleaseProgram(mProgram);
        oclSuccess("clReleaseProgram", this);
    }

    size_t* lSize = new size_t[mSource.size()];
    char** lSource = new char*[mSource.size()];
    for (unsigned int i=0; i<mSource.size(); i++)
    {
        lSource[i] = mSource[i].mText;
        lSize[i] = strlen(lSource[i]);
    }
    mProgram = clCreateProgramWithSource(mContext, mSource.size(), (const char **)lSource, lSize, &sStatusCL);
    delete [] lSource;
    delete [] lSize;
    if (!oclSuccess("clCreateProgramWithSource", this))
    {
        return false;
    }

    // always build program for all devices
    sStatusCL = clBuildProgram(mProgram, 0, 0, 0, NULL, NULL);
    if (!oclSuccess("clBuildProgram", this))
    {
        static char sBuffer[MAX_BUILD_LOG];
        size_t lSize = MAX_BUILD_LOG;
        sStatusCL = clGetProgramBuildInfo (mProgram, 
                                           mContext.getDevice(0), 
                                           CL_PROGRAM_BUILD_LOG, 
                                           lSize, 
                                           sBuffer, 
                                           &lSize);
        Log(KERNEL, this) << sBuffer;
        return false;
    }
    return true;
};

//
// Source Interface
//
char oclProgram::sRootPath[400];

void oclProgram::setRootPath(char* iRootPath)
{
    strncpy(sRootPath, iRootPath, 400);
}

void oclProgram::clrSource()
{
    for (unsigned int i=0; i<mSource.size(); i++)
    {
        srtSource& lSource = mSource[i];
        if (lSource.mPath[0]) // only delete if read from file
        {
            delete [] lSource.mText;
        }
    }

    mSource.clear();
}

void oclProgram::addSourceCode(char* iText)
{
    srtSource lSource;
    lSource.mText = iText;
    mSource.push_back(lSource);
}

void oclProgram::addSourceFile(char* iPath)
{
    char lPath[400];
    _getcwd(lPath, 400);

    _chdir(sRootPath);
    FILE* lFile = fopen(iPath, "rb");
    if (lFile)
    {
        srtSource lSource;
        strcpy(lSource.mPath, sRootPath);
        strcat(lSource.mPath, "\\");
        strcat(lSource.mPath, iPath);

        fseek(lFile, 0, SEEK_END);
        size_t lSize = ftell(lFile);
        lSource.mText = new char[lSize+1];
        fseek(lFile, 0, SEEK_SET);
        fread(lSource.mText, 1, lSize, lFile);
        fclose(lFile);

        lSource.mText[lSize] = '\0';
        mSource.push_back(lSource);
    }
    else Log(ERR, this) << "Unable to open source file " << iPath;

    _chdir(lPath);
}

char* oclProgram::getSourceCode(unsigned int iIndex)
{
    if (iIndex < mSource.size())
    {
        return mSource.at(iIndex).mText;
    }
    return 0;
}

char* oclProgram::getSourcePath(unsigned int iIndex)
{
    if (iIndex < mSource.size())
    {
        return mSource.at(iIndex).mPath;
    }
    return 0;
}

//
// Kernel Interface
//

cl_kernel oclProgram::createKernel(const char* iName)
{
    cl_kernel lKernel = clCreateKernel(mProgram, iName, &sStatusCL);
    oclSuccess("clCreateKernel", this);
    return lKernel;
}


void oclProgram::exportKernel(oclKernel& iKernel)
{
    iKernel.setOwner(this);
    mKernels.push_back(&iKernel);
};

oclKernel* oclProgram::getKernel(char* iName)
{
    for (unsigned int i=0; i<mKernels.size(); i++)
    {
        if (!strcmp(iName, mKernels[i]->getName()))
        {
            return mKernels[i];
        }
    }
    return 0;
};

//
//
//

void oclProgram::addEventHandler(srtEvent& iEvent)
{
    mEventHandler.push_back(&iEvent);
};

srtEvent* oclProgram::getEventHandler(char* iName)
{
    for (unsigned int i=0; i<mEventHandler.size(); i++)
    {
        if (!strcmp(iName, mEventHandler[i]->mName))
        {
            return mEventHandler[i];
        }
    }
    return 0;
};
