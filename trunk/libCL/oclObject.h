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
#ifndef _oclObject
#define _oclObject

#include <CL/cl.h>
#include <CL/cl_gl.h>

class oclObject
{
    public:

        oclObject(char* iName = "unnamed");
        virtual ~oclObject();

        void setName(char* iPtr);
        char* getName();

        void setData(void* iPtr);
        template <class TYPE> TYPE getData();

        void setOwner(void* iPtr);
        template <class TYPE> TYPE* getOwner();

        bool getError();
        void clrError();
        void setError();

    protected:

        void* mData;
        void* mOwner;
        char* mName;
        bool mError;

    private:

        oclObject(const oclObject&);
        oclObject& operator = (const oclObject&);

};

//
//
//

template <class TYPE> TYPE oclObject::getData() 
{
    return (TYPE)mData;
}

template <class TYPE> TYPE* oclObject::getOwner() 
{
    return (TYPE*)mOwner;
}


#endif