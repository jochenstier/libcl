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
        vector<srtSource>& getSource();

        static void setRootPath(char* iRootPath);

    protected:

        vector<srtSource> mSource;
        static char sRootPath[400];

    //
    // Kernel Interface
    //
    public:

        cl_kernel createKernel(const char* iName);

        // exported only
        vector<oclKernel*>& getKernels();
        oclKernel* getKernel(char* iName);

    protected:

        vector<oclKernel*> mKernels;
        void exportKernel(oclKernel& iKernel);
		void exportProgram(oclProgram& iProgram);
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
    srtEvent();
    srtEvent(char* iName, void* iData=0);

    virtual bool operator() (oclProgram&) = 0;

    template <class TYPE> TYPE getData();
    template <class TYPE> void setData(TYPE iData);

    void* mData;
    char* mName;
};

template <class TYPE> TYPE srtEvent::getData()
{
    return (TYPE) mData;
}

template <class TYPE> void srtEvent::setData(TYPE iData)
{
    mData = (void*) iData;
}

//
//
//

struct srtSource
{
    srtSource();

    char mPath[400];
    char* mText;
    size_t mLines;
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