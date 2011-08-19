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
#ifndef _oclRadixSort
#define _oclRadixSort

#include "oclProgram.h"
#include "oclBuffer.h"
#include "oclKernel.h"
#include "oclContext.h"

class oclRadixSort : public oclProgram
{
    public: 

	    oclRadixSort(oclContext& iContext);

		int compile();
        int compute(oclDevice& iDevice, oclBuffer& bfKey, oclBuffer& bfVal, int iStartBit, int iEndBit);

    protected:

		static const int cBits = 4;
		static const size_t cBlockSize = 256;
		static const size_t cMaxArraySize = cBlockSize*cBlockSize*4*cBlockSize/(1<<cBits);

		oclKernel clBlockSort;
		oclKernel clBlockScan;
		oclKernel clBlockPrefix;
		oclKernel clReorder;

		oclBuffer bfTempKey;
		oclBuffer bfTempVal;
		oclBuffer bfBlockScan;
		oclBuffer bfBlockSum;
		oclBuffer bfBlockOffset;

		void fit(oclBuffer& iBuffer, size_t iElements) ;

};      



#endif