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
#ifndef _oclRbm
#define _oclRbm

#include <vector>

#include "util/oclMemory.h"

class oclRbm : public oclProgram
{
    public: 

        oclRbm(oclContext& iContext, oclProgram* iParent = 0);
		~oclRbm();

        int compute(oclDevice& iDevice, oclImage2D& bfImage, cl_uint iEpochs);

        int getMap(oclDevice& iDevice, cl_uint iGibbs, cl_uint iMap, oclImage2D& bfImage);
		int getVis(oclDevice& iDevice, cl_uint iGibbs, oclImage2D& bfImage);
		int getImage(oclDevice& iDevice, oclImage2D& bfImage);

        int getWeight(oclDevice& iDevice, cl_uint iMap, oclImage2D& bfDest, cl_uint iDx, cl_uint iDy);

        int init(oclDevice& iDevice, cl_uint iInputW, cl_uint iInputH, cl_uint iMaps, cl_uint iKernelW, cl_uint iKernelH, cl_uint iPoolSw, cl_uint iPoolSh);

    protected:

        oclMemory mMemory;

        oclBuffer bfVis0;
        oclBuffer bfVisN;
		oclBuffer bfError;

		oclBuffer bfTTT;

        oclBuffer bfC;
		oclBuffer* bfDc[2];
        oclBuffer bfWtemp;

		cl_uint mInputW;
		cl_uint mInputH;

		typedef struct Map
		{
			oclBuffer bfHid0;
			oclBuffer bfHidN;
			oclBuffer bfPool;
			oclBuffer bfW;
			oclBuffer* bfdW[2];
			cl_uint mMapW;
			cl_uint mMapH;
			cl_uint mPoolSw;
			cl_uint mPoolSh;
			oclBuffer bfB;
			oclBuffer* bfDb[2];
			oclBuffer bfMean;

			Map(oclContext& iContext, char* iName)
			: bfHid0(iContext, iName)
			, bfHidN(iContext, iName)
			, bfPool(iContext, iName)
			, bfW(iContext, iName, oclBuffer::_float)
			, bfB(iContext, iName, oclBuffer::_float)
			, bfMean(iContext, iName, oclBuffer::_float)
			{
				bfdW[0] = new oclBuffer(iContext, iName, oclBuffer::_float);
				bfdW[1] = new oclBuffer(iContext, iName, oclBuffer::_float);
				bfDb[0] = new oclBuffer(iContext, iName, oclBuffer::_float);
				bfDb[1] = new oclBuffer(iContext, iName, oclBuffer::_float);
			}

			~Map()
			{
				bfMean.destroy();
				bfW.destroy();
				bfdW[0]->destroy();
				bfdW[1]->destroy();
				bfB.destroy();
				bfDb[0]->destroy();
				bfDb[1]->destroy();
				bfPool.destroy();
				bfHid0.destroy();
				bfHidN.destroy();
			}
			void create(int iMapW, int iMapH, int iKernelW, int iKernelH, int iPoolSw, int iPoolSh)
			{
				mMapW = iMapW;
				mMapH = iMapH;
				mPoolSw = iPoolSw;
				mPoolSh = iPoolSh;

			    bfHid0.create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, iMapW*iMapH);
			    bfHidN.create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, iMapW*iMapH);
				bfPool.create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mMapW/mPoolSw*mMapH/mPoolSh);
				bfW.create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, iKernelW*iKernelH);
				bfdW[0]->create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, iKernelW*iKernelH);
				bfdW[1]->create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, iKernelW*iKernelH);
				bfB.create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, iKernelW*iKernelH);
				bfDb[0]->create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 1);
				bfDb[1]->create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 1);
				bfMean.create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 1);
			}

		} Map;

        vector<Map*> bfMaps;

		oclBuffer bfTempA;
		oclBuffer bfBernoulli;

        oclKernel clGetMap;
        oclKernel clGetVis;
        oclKernel clGetWeight;

        oclKernel clLoadImage;
        oclKernel clGetImage;

        oclKernel clGibbsUpA;
        oclKernel clGibbsUpB;
        oclKernel clGibbsDnA;
        oclKernel clGibbsDnB0;
        oclKernel clGibbsDnB1;
        oclKernel clGibbsDnC;
        oclKernel clError;
		
	
        oclKernel clDwA;
        oclKernel clDwB;
        oclKernel clDc;
        oclKernel clDb;
        oclKernel clLearn;

        oclKernel clSparsity;

		cl_uint mKernelW;
		cl_uint mKernelH;
		cl_float mRate;
		cl_float mMomentum;
		cl_float mPenalty;
		cl_float mSparsity;

		double sum(oclBuffer& iBuffer);

		bool gibbsSampling(oclDevice& iDevice);
};      


#endif
