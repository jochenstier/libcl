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
#include "oclObject.h"


oclObject::oclObject(char* iName)
: mName(iName)
, mData(0)
, mOwner(0)
, mError(0)
{
}

oclObject::~oclObject()
{
};

void oclObject::setData(void* iData)
{
    mData = iData;
};

void oclObject::setOwner(void* iOwner)
{
    mOwner = iOwner;
};

void oclObject::setName(char* iData)
{
    mName = iData;
}

char* oclObject::getName()
{
    return mName;
}

//
//
//



void oclObject::setError()
{
    mError = true;
}

void oclObject::clrError()
{
    mError = false;
}

bool oclObject::getError()
{
    return mError;
}
