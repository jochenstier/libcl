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
#ifndef _oclProgram
#define _oclProgram

#include <vector>
#include <functional>

#include "oclContext.h"
#include "oclKernel.h"

struct srtEvent;
struct srtSource;

class oclProgram : public oclObject
{
    static const int MAX_BUILD_LOG = 30000;

    public: 

        oclProgram(oclContext& iContext, char* iName);
        ~oclProgram();

        operator cl_program ();
        oclContext& getContext();

        virtual int compile();

    protected:

        cl_program mProgram;
        oclContext& mContext;

    private:

        oclProgram(const oclProgram&);
        oclProgram& operator = (const oclProgram&);

    //
    // Source Interface
    //
    public:

        void clrSource();
        void addSourceCode(char* iCode);
        void addSourceFile(char* iPath);
        char* getSourceCode(unsigned int iIndex);
        char* getSourcePath(unsigned int iIndex);

        static void setRootPath(char* iRootPath);

    protected:

        vector<srtSource> mSource;
        static char sRootPath[400];


    //
    // Kernel Interface
    //
    public:

        oclKernel* getKernel(char* iName);
        cl_kernel createKernel(const char* iName);

    protected:

        void exportKernel(oclKernel& iKernel);

    protected:

        vector<oclKernel*> mKernels;

    //
    // Event Interface
    //
    public:

        virtual void addEventHandler(srtEvent& iEvent);
        srtEvent* getEventHandler(char* iName);

    protected:

        vector<srtEvent*> mEventHandler;

};      

//
//
//

struct srtEvent
{
    srtEvent()
    : mName("")
    , mData(0)
    {
    }
    srtEvent(char* iName, void* iIndex=0)
    : mName(iName)
    , mData(iIndex)
    {
    }

    virtual bool operator() (oclProgram&) = 0;

    template <class TYPE> TYPE getData()
    {
        return (TYPE) mData;
    }
    template <class TYPE> void setData(TYPE iData)
    {
        mData = (void*) iData;
    }

    void* mData;
    char* mName;
};

//
//
//

struct srtSource
{
    char mPath[400];
    char* mText;

    srtSource()
    : mText(0)
    {
        mPath[0] = 0;
    }
};


//
//
//

#define ENQUEUE_VALIDATE \
if (!oclSuccess("clEnqueueNDRangeKernel", this))\
{\
    return false;\
}\

#endif