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
#include <math.h>
#include "oclRbm.h"

static const int CONV_RADIUS = 7;

oclRbm::oclRbm(oclContext& iContext, oclProgram* iParent)
: oclProgram(iContext, "oclRbm", iParent)
// buffers
, bfVis0(iContext, "bfVis0")
, bfVisN(iContext, "bfVisN")
, bfError(iContext, "bfError", oclBuffer::_float)
, bfC(iContext, "bfC")
, bfWtemp(iContext, "bfWtemp")

, bfTempA(iContext, "bfTempA", oclBuffer::_float)
, bfTTT(iContext, "bfTTT", oclBuffer::_float)

, bfMaps(0)
, bfBernoulli(iContext, "bfBernoulli", oclBuffer::_float)
// kernels
, clGetMap(*this, "clGetMap")
, clGetVis(*this, "clGetVis")
, clGetWeight(*this, "clGetWeight")

, clLoadImage(*this, "clLoadImage")
, clGetImage(*this, "clGetImage")

, clGibbsUpA(*this, "clGibbsUpA")
, clGibbsUpB(*this, "clGibbsUpB")
, clGibbsDnA(*this, "clGibbsDnA")
, clGibbsDnB0(*this, "clGibbsDnB0")
, clGibbsDnB1(*this, "clGibbsDnB1")
, clGibbsDnC(*this, "clGibbsDnC")
, clError(*this, "clError")

, clDwA(*this, "clDwA")
, clDwB(*this, "clDwB")
, clDc(*this, "clDc")
, clDb(*this, "clDb")
, clLearn(*this, "clLearn")

, clSparsity(*this, "clSparsity")

// programs
, mMemory(iContext, this)

, mKernelW(5)
, mKernelH(5)
, mInputW(0) 
, mInputH(0)

, mRate(0.01f)
, mMomentum(0.0f) 
, mPenalty(0.05f)
, mSparsity(0.02f)
{
    //cl_image_format lFormat = { CL_RGBA,  CL_HALF_FLOAT };
    bfVis0.create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 256 * 256);
    bfVisN.create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 256 * 256);
    bfBernoulli.create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 256 * 256);
	bfError.create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 1);
	bfTTT.create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 1);
	bfWtemp.create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 1);

    bfC.create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 1);
	bfDc[0] = new oclBuffer(iContext, "bfDc0");
	bfDc[0]->create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 1);
	bfDc[1] = new oclBuffer(iContext, "bfDc1");
	bfDc[1]->create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 1);

	bfTempA.create<cl_float>(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mKernelW*mKernelH);

    addSourceFile("learning/oclRbm.cl");
}

oclRbm::~oclRbm()
{
	delete bfDc[0];
	delete bfDc[1];

	for(vector<Map*>::iterator lIter = bfMaps.begin();  lIter != bfMaps.end(); lIter++)
	{ 
		delete *lIter;
	}
}

int oclRbm::init(oclDevice& iDevice, cl_uint iInputW, cl_uint iInputH, cl_uint iMaps, cl_uint iKernelW, cl_uint iKernelH, cl_uint iPoolSw, cl_uint iPoolSh)
{
	mKernelW = iKernelW;
	mKernelH = iKernelH;
	mInputW = iInputW;
	mInputH = iInputH;
    int lMapW = iInputW - mKernelW+1;
    int lMapH = iInputH - mKernelH+1;

    bfVis0.resize<float>(mInputW * mInputH);
    bfVisN.resize<float>(mInputW * mInputH);
	bfC.resize<float>(1);
	bfTempA.resize<float>(mInputW*mInputH*mKernelW*mKernelH);
    bfBernoulli.resize<float>(lMapW*lMapH);
	bfWtemp.resize<cl_float>(lMapW*lMapH);

    mMemory.memSet(iDevice, bfC, 0);
    mMemory.memSet(iDevice, *bfDc[0], 0.0f);
    mMemory.memSet(iDevice, *bfDc[1], 0.0f);
srand(10);
	float lRange = sqrt(6.0f/(iInputW*iInputH*iMaps));

	Log(INFO) << "Range = " << lRange;
	for (cl_uint i=0; i<iMaps; i++)
	{
		char lName[100];
		sprintf(lName, "bfMap%d", i);	 
		Map* lMap = new Map(mContext, lName); 
	    lMap->create(lMapW, lMapH, mKernelW, mKernelH, iPoolSw, iPoolSh);
		bfMaps.push_back(lMap);

		mMemory.random(iDevice, lMap->bfW, -lRange, lRange);
	    mMemory.memSet(iDevice, *lMap->bfdW[0], 0.0f);
	    mMemory.memSet(iDevice, *lMap->bfdW[1], 0.0f);
		
		mMemory.memSet(iDevice, lMap->bfB, 0.0);
	    mMemory.memSet(iDevice, *lMap->bfDb[0], 0.0f);
	    mMemory.memSet(iDevice, *lMap->bfDb[1], 0.0f);
	}

	return 1;
}

int oclRbm::getMap(oclDevice& iDevice, cl_uint iGibbs, cl_uint iLayer, oclImage2D& bfImage)
{
    size_t lGlobalSize[2];
    lGlobalSize[0] = bfMaps[iLayer]->mMapW;
    lGlobalSize[1] = bfMaps[iLayer]->mMapH;
	
	clSetKernelArg(clGetMap, 0, sizeof(cl_mem), bfImage);
	clSetKernelArg(clGetMap, 1, sizeof(cl_mem), iGibbs == 0 ? bfMaps[iLayer]->bfHid0 : bfMaps[iLayer]->bfHidN);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clGetMap, 2, NULL, lGlobalSize, NULL, 0, NULL, clGetMap.getEvent());
	VALIDATE_KENEL(clGetMap)

	return 1;
}

int oclRbm::getImage(oclDevice& iDevice, oclImage2D& bfImage)
{
    size_t lGlobalSize[2];
    lGlobalSize[0] = mInputW;
    lGlobalSize[1] = mInputH;
	clSetKernelArg(clGetImage, 0, sizeof(cl_mem), bfImage);
	clSetKernelArg(clGetImage, 1, sizeof(cl_mem), bfVisN);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clGetImage, 2, NULL, lGlobalSize, NULL, 0, NULL, clGetImage.getEvent());
	VALIDATE_KENEL(clGetImage)
	return 1;
}
int oclRbm::getVis(oclDevice& iDevice, cl_uint iGibbs, oclImage2D& bfImage)
{
    size_t lGlobalSize[2];
    lGlobalSize[0] = mInputW;
    lGlobalSize[1] = mInputH;
	clSetKernelArg(clGetVis, 0, sizeof(cl_mem), bfImage);
	clSetKernelArg(clGetVis, 1, sizeof(cl_mem), iGibbs == 0 ? bfVis0 : bfVisN);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clGetVis, 2, NULL, lGlobalSize, NULL, 0, NULL, clGetVis.getEvent());
	VALIDATE_KENEL(clGetVis)
	return 1;
}



int oclRbm::getWeight(oclDevice& iDevice, cl_uint iMap, oclImage2D& bfDest, cl_uint iDx, cl_uint iDy)
{
	mMemory.normalize(iDevice, bfMaps[iMap]->bfW, bfWtemp);

    size_t lGlobalSize[2];
    lGlobalSize[0] = mKernelW;
    lGlobalSize[1] = mKernelH;
	clSetKernelArg(clGetWeight, 0, sizeof(cl_mem), bfDest);
	clSetKernelArg(clGetWeight, 1, sizeof(cl_mem), bfWtemp);
	clSetKernelArg(clGetWeight, 2, sizeof(cl_int), &iDx);
	clSetKernelArg(clGetWeight, 3, sizeof(cl_int), &iDy);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clGetWeight, 2, NULL, lGlobalSize, NULL, 0, NULL, clGetWeight.getEvent());
	VALIDATE_KENEL(clGetWeight)

	return 1;
};


int oclRbm::compute(oclDevice& iDevice, oclImage2D& bfImage, cl_uint iEpochs)
{
    size_t lGlobalSize[2];
	float lZero = 0;

	// compute epochs of training set
	for (int e=0; e<1; e++)
	{
		//
		// X1
		//
		lGlobalSize[0] = mInputW;
		lGlobalSize[1] = mInputH;
		clSetKernelArg(clLoadImage, 0, sizeof(cl_mem), bfImage);
		clSetKernelArg(clLoadImage, 1, sizeof(cl_mem), bfVis0);
		sStatusCL = clEnqueueNDRangeKernel(iDevice, clLoadImage, 2, NULL, lGlobalSize, NULL, 0, NULL, clLoadImage.getEvent());
		VALIDATE_KENEL(clLoadImage)

		mMemory.normalize(iDevice, bfVis0, bfVis0);
		
		gibbsSampling(iDevice);

		// compute Error
		lGlobalSize[0] = 1;
		lGlobalSize[1] = 1;
		clSetKernelArg(clError, 0, sizeof(cl_mem), bfVis0);
		clSetKernelArg(clError, 1, sizeof(cl_mem), bfVisN);
		clSetKernelArg(clError, 2, sizeof(cl_mem), bfError);
		clSetKernelArg(clError, 3, sizeof(cl_int), &mInputW);
		clSetKernelArg(clError, 4, sizeof(cl_int), &mInputH);
		sStatusCL = clEnqueueNDRangeKernel(iDevice, clError, 2, NULL, lGlobalSize, NULL, 0, NULL, clError.getEvent());
		VALIDATE_KENEL(clError) 
		Log(INFO) << "************* bfError " << sum(bfError);

		// 
		// Contrastive divergence
		// 
		for(vector<Map*>::iterator lIter = bfMaps.begin();  lIter != bfMaps.end(); lIter++)
		{
			//
			// update weights
			//
			lGlobalSize[0] = (*lIter)->mMapW;
			lGlobalSize[1] = (*lIter)->mMapH;
			clSetKernelArg(clDwA, 0, sizeof(cl_mem), bfVis0);
			clSetKernelArg(clDwA, 1, sizeof(cl_mem), (*lIter)->bfHid0);
			clSetKernelArg(clDwA, 2, sizeof(cl_mem), bfVisN);
			clSetKernelArg(clDwA, 3, sizeof(cl_mem), (*lIter)->bfHidN); 
			clSetKernelArg(clDwA, 4, sizeof(cl_mem), bfTempA);
			clSetKernelArg(clDwA, 5, sizeof(cl_int), &mKernelW);
			clSetKernelArg(clDwA, 6, sizeof(cl_int), &mKernelH);
			sStatusCL = clEnqueueNDRangeKernel(iDevice, clDwA, 2, NULL, lGlobalSize, NULL, 0, NULL, clDwA.getEvent());
			VALIDATE_KENEL(clDwA) 

			lGlobalSize[0] = mKernelW;
			lGlobalSize[1] = mKernelH;
			clSetKernelArg(clDwB, 0, sizeof(cl_mem), *((*lIter)->bfdW[(e+1)%2]));
			clSetKernelArg(clDwB, 1, sizeof(cl_mem), bfTempA);
			clSetKernelArg(clDwB, 2, sizeof(cl_int), &(*lIter)->mMapW);
			clSetKernelArg(clDwB, 3, sizeof(cl_int), &(*lIter)->mMapH);
			sStatusCL = clEnqueueNDRangeKernel(iDevice, clDwB, 2, NULL, lGlobalSize, NULL, 0, NULL, clDwB.getEvent());
			VALIDATE_KENEL(clDwB) 

			lGlobalSize[0] = mKernelW*mKernelH;
			clSetKernelArg(clLearn, 0, sizeof(cl_mem), (*lIter)->bfW);
			clSetKernelArg(clLearn, 1, sizeof(cl_mem), *((*lIter)->bfdW[(e+1)%2]));
			clSetKernelArg(clLearn, 2, sizeof(cl_mem), *((*lIter)->bfdW[e%2]));
			clSetKernelArg(clLearn, 3, sizeof(cl_float), &mMomentum);
			clSetKernelArg(clLearn, 4, sizeof(cl_float), &mPenalty); 
			clSetKernelArg(clLearn, 5, sizeof(cl_float), &mRate);
			sStatusCL = clEnqueueNDRangeKernel(iDevice, clLearn, 1, NULL, lGlobalSize, NULL, 0, NULL, clLearn.getEvent());
			VALIDATE_KENEL(clLearn)

			//
			// update B
			//

			// compute dB
			lGlobalSize[0] = 1;
			lGlobalSize[1] = 1;
			clSetKernelArg(clDb, 0, sizeof(cl_mem), (*lIter)->bfHid0);
			clSetKernelArg(clDb, 1, sizeof(cl_mem), (*lIter)->bfHidN); 
			clSetKernelArg(clDb, 2, sizeof(cl_mem), *((*lIter)->bfDb[(e+1)%2]));
			clSetKernelArg(clDb, 3, sizeof(cl_int), &(*lIter)->mMapW);
			clSetKernelArg(clDb, 4, sizeof(cl_int), &(*lIter)->mMapH);
			clSetKernelArg(clDb, 5, sizeof(cl_int), &mKernelW);
			clSetKernelArg(clDb, 6, sizeof(cl_int), &mKernelH);
			sStatusCL = clEnqueueNDRangeKernel(iDevice, clDb, 2, NULL, lGlobalSize, NULL, 0, NULL, clDb.getEvent());
			VALIDATE_KENEL(clDb) 

			//Log(INFO) << "(*lIter)->bfdB[(e+1)%2]) " << sum(*((*lIter)->bfDb[(e+1)%2]));

			// learn B
			lGlobalSize[0] = 1;
			clSetKernelArg(clLearn, 0, sizeof(cl_mem), (*lIter)->bfB);
			clSetKernelArg(clLearn, 1, sizeof(cl_mem), *((*lIter)->bfDb[(e+1)%2]));
			clSetKernelArg(clLearn, 2, sizeof(cl_mem), *((*lIter)->bfDb[e%2]));
			clSetKernelArg(clLearn, 3, sizeof(cl_float), &mMomentum);
			clSetKernelArg(clLearn, 4, sizeof(cl_float), &lZero); 
			clSetKernelArg(clLearn, 5, sizeof(cl_float), &mRate);
			sStatusCL = clEnqueueNDRangeKernel(iDevice, clLearn, 1, NULL, lGlobalSize, NULL, 0, NULL, clLearn.getEvent());
			VALIDATE_KENEL(clLearn)

			//Log(INFO) << "(*lIter)->bfB AFTER LEARN " << sum((*lIter)->bfB);

			// compute Mean
			mMemory.mean(iDevice, (*lIter)->bfHid0, (*lIter)->bfMean);

			//Log(INFO) << "(*lIter)->bfMean " << sum((*lIter)->bfMean);

			// apply Sparsity
			lGlobalSize[0] = 1;
			lGlobalSize[1] = 1;
			clSetKernelArg(clSparsity, 0, sizeof(cl_mem), (*lIter)->bfB);
			clSetKernelArg(clSparsity, 1, sizeof(cl_mem), (*lIter)->bfMean);
			clSetKernelArg(clSparsity, 2, sizeof(cl_float), &mSparsity);
			clSetKernelArg(clSparsity, 3, sizeof(cl_float), &mRate);
			sStatusCL = clEnqueueNDRangeKernel(iDevice, clSparsity, 2, NULL, lGlobalSize, NULL, 0, NULL, clSparsity.getEvent());
			VALIDATE_KENEL(clSparsity) 
//
//			Log(INFO) << "(*lIter)->bfB " << sum((*lIter)->bfB);

		}
		
		//
		// update C
		//

		//Log(INFO) << "(*lIter)->bfC BEFORE LEARN " << sum(bfC);

		lGlobalSize[0] = 1;
		lGlobalSize[1] = 1;
		int lMaps = bfMaps.size();
		clSetKernelArg(clDc, 0, sizeof(cl_mem), bfVis0);
		clSetKernelArg(clDc, 1, sizeof(cl_mem), bfVisN); 
		clSetKernelArg(clDc, 2, sizeof(cl_mem), (*bfDc[(e+1)%2]));
		clSetKernelArg(clDc, 3, sizeof(cl_int), &mInputW);
		clSetKernelArg(clDc, 4, sizeof(cl_int), &mInputH);
		clSetKernelArg(clDc, 5, sizeof(cl_int), &mKernelW);
		clSetKernelArg(clDc, 6, sizeof(cl_int), &mKernelH);
		clSetKernelArg(clDc, 7, sizeof(cl_int), &lMaps);
		sStatusCL = clEnqueueNDRangeKernel(iDevice, clDc, 2, NULL, lGlobalSize, NULL, 0, NULL, clDc.getEvent());
		VALIDATE_KENEL(clDc) 

		//Log(INFO) << "(*lIter)->bfDc[(e+1)%2]) " << sum(*bfDc[(e+1)%2]); 

		lGlobalSize[0] = 1;
		clSetKernelArg(clLearn, 0, sizeof(cl_mem), bfC);
		clSetKernelArg(clLearn, 1, sizeof(cl_mem), (*bfDc[(e+1)%2]));
		clSetKernelArg(clLearn, 2, sizeof(cl_mem), (*bfDc[e%2]));
		clSetKernelArg(clLearn, 3, sizeof(cl_float), &mMomentum);
		clSetKernelArg(clLearn, 4, sizeof(cl_float), &lZero); 
		clSetKernelArg(clLearn, 5, sizeof(cl_float), &mRate);
		sStatusCL = clEnqueueNDRangeKernel(iDevice, clLearn, 1, NULL, lGlobalSize, NULL, 0, NULL, clLearn.getEvent());
		VALIDATE_KENEL(clLearn)

		//Log(INFO) << "(*lIter)->bfC " << sum(bfC);

	}

    return true;
}


bool oclRbm::gibbsSampling(oclDevice& iDevice)
{
    size_t lGlobalSize[2];  

	// UP
	for(vector<Map*>::iterator lIter = bfMaps.begin();  lIter != bfMaps.end(); lIter++)
	{ 
		// upward pass
		lGlobalSize[0] = (*lIter)->mMapW;
		lGlobalSize[1] = (*lIter)->mMapH;
 
		clSetKernelArg(clGibbsUpA, 0, sizeof(cl_mem), bfVis0);
		clSetKernelArg(clGibbsUpA, 1, sizeof(cl_mem), bfTempA);
		clSetKernelArg(clGibbsUpA, 2, sizeof(cl_mem), (*lIter)->bfW);
		clSetKernelArg(clGibbsUpA, 3, sizeof(cl_mem), (*lIter)->bfB);
		clSetKernelArg(clGibbsUpA, 4, sizeof(cl_int), &mKernelW);
		clSetKernelArg(clGibbsUpA, 5, sizeof(cl_int), &mKernelH);
		sStatusCL = clEnqueueNDRangeKernel(iDevice, clGibbsUpA, 2, NULL, lGlobalSize, NULL, 0, NULL, clGibbsUpA.getEvent());
		VALIDATE_KENEL(clGibbsUpA)

		clSetKernelArg(clGibbsUpB, 0, sizeof(cl_mem), bfTempA);
		clSetKernelArg(clGibbsUpB, 1, sizeof(cl_mem), (*lIter)->bfHid0);
		clSetKernelArg(clGibbsUpB, 2, sizeof(cl_int), &(*lIter)->mPoolSw);
		clSetKernelArg(clGibbsUpB, 3, sizeof(cl_int), &(*lIter)->mPoolSh);
		sStatusCL = clEnqueueNDRangeKernel(iDevice, clGibbsUpB, 2, NULL, lGlobalSize, NULL, 0, NULL, clGibbsUpB.getEvent());
		VALIDATE_KENEL(clGibbsUpB)

		//Log(INFO) << "bfVis0  " << sum(bfVis0);
		//Log(INFO) << "(*lIter)->bfW  " << sum((*lIter)->bfW);
		//Log(INFO) << "(*lIter)->bfHid0  " << sum((*lIter)->bfHid0);
	}

	// Down
	lGlobalSize[0] = mInputW; 
	lGlobalSize[1] = mInputH;
	clSetKernelArg(clGibbsDnA, 0, sizeof(cl_mem), bfVisN);
	clSetKernelArg(clGibbsDnA, 1, sizeof(cl_mem), bfC);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clGibbsDnA, 2, NULL, lGlobalSize, NULL, 0, NULL, clGibbsDnA.getEvent());
	VALIDATE_KENEL(clGibbsDnA)
	for(vector<Map*>::iterator lIter = bfMaps.begin();  lIter != bfMaps.end(); lIter++)
	{
		mMemory.random(iDevice, bfBernoulli, 0,1);
		mMemory.memSet(iDevice, bfTempA, 0.0f);

		lGlobalSize[0] = (*lIter)->mMapW;
		lGlobalSize[1] = (*lIter)->mMapH;
		clSetKernelArg(clGibbsDnB0, 0, sizeof(cl_mem), bfTempA);
		clSetKernelArg(clGibbsDnB0, 1, sizeof(cl_mem), (*lIter)->bfHid0);
		clSetKernelArg(clGibbsDnB0, 2, sizeof(cl_mem), bfBernoulli);
		clSetKernelArg(clGibbsDnB0, 3, sizeof(cl_mem), (*lIter)->bfW);
		clSetKernelArg(clGibbsDnB0, 4, sizeof(cl_int), &mKernelW);
		clSetKernelArg(clGibbsDnB0, 5, sizeof(cl_int), &mKernelH);
		sStatusCL = clEnqueueNDRangeKernel(iDevice, clGibbsDnB0, 2, NULL, lGlobalSize, NULL, 0, NULL, clGibbsDnB0.getEvent());
		VALIDATE_KENEL(clGibbsDnB0)

		lGlobalSize[0] = mInputW; 
		lGlobalSize[1] = mInputH;
		clSetKernelArg(clGibbsDnB1, 0, sizeof(cl_mem), bfVisN);
		clSetKernelArg(clGibbsDnB1, 1, sizeof(cl_mem), bfTempA);
		clSetKernelArg(clGibbsDnB1, 2, sizeof(cl_int), &mKernelW);
		clSetKernelArg(clGibbsDnB1, 3, sizeof(cl_int), &mKernelH);
		sStatusCL = clEnqueueNDRangeKernel(iDevice, clGibbsDnB1, 2, NULL, lGlobalSize, NULL, 0, NULL, clGibbsDnB1.getEvent());
		VALIDATE_KENEL(clGibbsDnB1)

	}
	clSetKernelArg(clGibbsDnC, 0, sizeof(cl_mem), bfVisN);
	clSetKernelArg(clGibbsDnC, 1, sizeof(cl_mem), bfVis0);
	sStatusCL = clEnqueueNDRangeKernel(iDevice, clGibbsDnC, 2, NULL, lGlobalSize, NULL, 0, NULL, clGibbsDnC.getEvent());
	VALIDATE_KENEL(clGibbsDnC)

	mMemory.normalize(iDevice, bfVisN, bfVisN);

	// UP
	for(vector<Map*>::iterator lIter = bfMaps.begin();  lIter != bfMaps.end(); lIter++)
	{
		// upward pass
		lGlobalSize[0] = (*lIter)->mMapW;
		lGlobalSize[1] = (*lIter)->mMapH;

		clSetKernelArg(clGibbsUpA, 0, sizeof(cl_mem), bfVisN);
		clSetKernelArg(clGibbsUpA, 1, sizeof(cl_mem), bfTempA);
		clSetKernelArg(clGibbsUpA, 2, sizeof(cl_mem), (*lIter)->bfW);
		clSetKernelArg(clGibbsUpA, 3, sizeof(cl_mem), (*lIter)->bfB);
		clSetKernelArg(clGibbsUpA, 4, sizeof(cl_int), &mKernelW);
		clSetKernelArg(clGibbsUpA, 5, sizeof(cl_int), &mKernelH);
		sStatusCL = clEnqueueNDRangeKernel(iDevice, clGibbsUpA, 2, NULL, lGlobalSize, NULL, 0, NULL, clGibbsUpA.getEvent());
		VALIDATE_KENEL(clGibbsUpA)
		//Log(INFO) << "(*lIter)->bfHidN  " << sum((*lIter)->bfHidN);

		clSetKernelArg(clGibbsUpB, 0, sizeof(cl_mem), bfTempA);
		clSetKernelArg(clGibbsUpB, 1, sizeof(cl_mem), (*lIter)->bfHidN);
		clSetKernelArg(clGibbsUpB, 2, sizeof(cl_int), &(*lIter)->mPoolSw);
		clSetKernelArg(clGibbsUpB, 3, sizeof(cl_int), &(*lIter)->mPoolSh);
		sStatusCL = clEnqueueNDRangeKernel(iDevice, clGibbsUpB, 2, NULL, lGlobalSize, NULL, 0, NULL, clGibbsUpB.getEvent());
		VALIDATE_KENEL(clGibbsUpB)
	} 
	return true;
}


double oclRbm::sum(oclBuffer& iBuffer)
{
	double lSum = 0;
	if (iBuffer.map(CL_MAP_READ))
	{
		int lDim = iBuffer.count<cl_float>();
		cl_float* lPtr = iBuffer.ptr<cl_float>();
		for (int i=0; i<lDim; i++)
		{
			lSum += lPtr[i];
		}
		iBuffer.unmap();
	}
	return lSum;
}