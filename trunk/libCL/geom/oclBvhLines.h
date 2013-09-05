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
#ifndef _oclBvhLines
#define _oclBvhLines

#include "oclProgram.h"

#include "sort\\oclRadixSort.h"

class oclBvhLines : public oclProgram
{
    public: 

	    oclBvhLines(oclContext& iContext, oclProgram* iParent = 0);
       ~oclBvhLines();

        int compute(oclDevice& iDevice, oclBuffer& bfVertex);

		typedef struct 
		{
			cl_float2 bbMin;
			cl_float2 bbMax;
			cl_uint left;
			cl_uint right;
			cl_uint bit;
			cl_uint trav;
		} BVHNode;

    	oclBuffer& getNodeBuffer();
		cl_uint getNodeCount();
		cl_uint getRootNode();

    protected:

		oclRadixSort mRadixSort;

        void create();
        void destroy();

		typedef struct 
		{
			cl_float2 bbMin;
			cl_float2 bbMax;
			cl_int flag;
		} 
        srtAABB;

        static const int sBVH	= 0x001;

		static const cl_uint cWarpSize = 32;

		oclKernel clMorton;
		oclKernel clCreateNodes;
		oclKernel clLinkNodes;
		oclKernel clCreateLeaves;
		oclKernel clComputeAABBs;

		oclBuffer bfMortonKey;
		oclBuffer bfMortonVal;
		oclBuffer bfBvhRoot;
		oclBuffer bfBvhNode;

		cl_uint mRootNode;
};      

#endif
