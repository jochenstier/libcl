// Copyright [2013] [Geist Software Labs Inc.] All rights reserved

//
// Contrastive Divergence
//

__kernel void clLoadImage(__read_only image2d_t image, __global float* lvis0)
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int w = get_global_size(0);

    float4 img = read_imagef(image, sampler, (float2)(x,y)); 

    lvis0[y*w+x] = 0.21*img.x + 0.72*img.y + 0.07*img.z;  // x(1)
}


__kernel void clGibbsUpA(__global const float* lvis, __global float* ltemp, __global const float* weight, __global const float* B,  int kW, int kH)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int w = get_global_size(0);

    const int visW = w+kW-1;

    float v = B[0];
    for (int j=0; j<kH; j++)
    {
	  for (int i=0; i<kW; i++)
	  {
		v += lvis[(y+j)*visW+x+i]*weight[j*kW+i];
       }
    }

    ltemp[y*w+x] = v;
    //ltemp[y*w+x] = exp(v);  //probabilistic max pooling
}

__kernel void clGibbsUpB(__global const float* ltemp, __global float* lhid,  int sW, int sH)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int w = get_global_size(0);

    const int bx = x/sW;
    const int by = y/sH;

    float v = 0;
    for (int j=0; j<sH; j++)
    {
	  for (int i=0; i<sW; i++)
	  {
		v += ltemp[(by*sH+j)*w+bx*sW+i];
        }
    }
    lhid[y*w+x] = 1/(1+exp(-ltemp[y*w+x])); // y(1)
 //   lhid[y*w+x] = ltemp[y*w+x];///(1+v); // y(1) probabilistic max pooling
}



__kernel void clGibbsDnA(__global float* lvis, __global const float* C)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int w = get_global_size(0);

    lvis[y*w+x] = C[0]; 
}

__kernel void clGibbsDnB0(__global float* ltemp, __global const float* lhid, __global const float* lbernoulli, __global const float* weight, int kW, int kH)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int w = get_global_size(0);
    const int h = get_global_size(1);

    const int visW = w+kW-1;
    const int visH = h+kH-1;
    const int stride = visW*visH;

    float p = lhid[y*w+x];
/*
    for (int j=0; j<kH; j++)
    {
	  for (int i=0; i<kW; i++)
	  {
           int wIndex = j*kW+i;
		ltemp[stride*wIndex + wIndex] = weight[wIndex]*p;//(p > lbernoulli[y*w+x] ? 1 : 0);
		//ltemp[stride*wIndex + wIndex] = weight[wIndex]*(p > 0.5 ? 1 : 0);
		//ltemp[stride*wIndex + wIndex] = weight[wIndex]*p;
        }
    }
*/
    for (int j=0; j<kH; j++)
    {
	  for (int i=0; i<kW; i++)
	  {
		int visI = (y+j)*visW+x+i;
		int bucket = kH*kW*visI;
		int wIndex = j*kW+i;
		ltemp[bucket + wIndex] = weight[wIndex]*(p > lbernoulli[y*w+x] ? 1 : 0);
        }
    }

}

__kernel void clGibbsDnB1(__global float* lvis, __global const float* ltemp, int kW, int kH)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int w = get_global_size(0);
    const int h = get_global_size(1);

    const int stride = w*h;
/*
    float v =  lvis[y*w+x];
    for (int j=0; j<kH; j++)
    {
	  for (int i=0; i<kW; i++)
	  {
           int wIndex = j*kW+i;
		v += ltemp[stride*wIndex+wIndex];
	  }
    }
    lvis[y*w+x] = v;
*/
    int bucket = kW*kH*(y*w+x);

    float v =  lvis[y*w+x];
    for (int j=0; j<kH; j++)
    {
	  for (int i=0; i<kW; i++)
	  {
           int wIndex = j*kW+i;
		v += ltemp[bucket+wIndex];
	  }
    }
    lvis[y*w+x] = v;
}


__kernel void clGibbsDnC(__global float* lvisN, __global float* lvis0)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int w = get_global_size(0);

    float sigma = 1.0;
    float v = lvis0[y*w+x] - lvisN[y*w+x];
    lvisN[y*w+x] = 1.0f/(sigma*sqrt(2*M_PI))*exp(-(v*v)/(2*sigma*sigma));

/*
    float sigma = 1.0;
    float v = lvisN[y*w+x];
    lvisN[y*w+x] = 1.0f/(sigma*sqrt(2*M_PI))*exp(-(v*v)/(2*sigma*sigma));
*/
}

//
// Compute Gradients
//

__kernel void clDwA(__global const float* lvis0, __global const float* lhid0, __global const float* lvisN, __global const float* lhidN, __global float* corr, int kW, int kH)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int w = get_global_size(0);
    const int h = get_global_size(1);

    const int visW = w+kW-1;
    const int visH = h+kH-1;
    const int stride = w*h;

    int ihid= y*w+x;
    float hid0 = lhid0[ihid];
    float hidN = lhidN[ihid];
    for (int j=0; j<kH; j++)
    {
	  for (int i=0; i<kW; i++)
	  {
           int wIndex = j*kW+i;
		int ivis= (y+j)*visW+x+i;
		corr[stride*wIndex + ihid] = lvis0[ivis]*hid0 - lvisN[ivis]*hidN;
        }
    }
}

__kernel void clDwB( __global float* dw, __global float* corr,int mW, int mH)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int w = get_global_size(0);
    const int h = get_global_size(1);

    const int stride = mW*mH ;

    float v = 0;
    int wIndex= y*w+x;
    for (int j=0; j<mH; j++)
    {
	  for (int i=0; i<mW; i++)
	  {
	       int ihid= j*mW+i;
	       v += corr[wIndex*stride+ihid];
	  }
    }
    dw[y*w+x] = v/(mH*mW);
}


__kernel void clDc(__global const float* lvis0, __global const float* lvisN, __global float* dC, int vW, int vH, int kW, int kH, int maps)
{
    float v = 0;
    for (int j=0; j<vH*vW; j++)
    {
	  v += lvis0[j] - lvisN[j];
    }
    dC[0] = v/(vW*vH); 
}


__kernel void clDb(__global const float* lhid0, __global const float* lhidN, __global float* dB, int mW, int mH, int kW, int kH)
{
    float v = 0;
    for (int j=0; j<mH*mW; j++)
    {
	  v += lhid0[j] - lhidN[j];
    }

    dB[0] = v/(mW*mH);
}


__kernel void clLearn(__global float* p, __global const float* dnext, __global const float* dprev, float momentum, float penalty, float rate)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int w = get_global_size(0);

    const int index = y*w+x;    

    float grad = momentum*dprev[index] + (1.0-momentum)*dnext[index];
    p[index] = p[index] + rate*(grad - penalty*p[index]);
}



//
// Sparsity
//

__kernel void clSparsity(__global float* B, __global const float* mean, float sparsity, float rate)
{
	//dcSparse = self.lRate*self.sparseGain*(squeeze(self.sparsity -mean(mean(self.eHid0))));
	//self.c = self.c + dcSparse;

	B[0] = B[0] + rate*(sparsity - mean[0]);
}




//
// Error
//

__kernel void clError(__global const float* lvis0, __global const float* lvisN, __global float* error, int mW, int mH)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    float v = 0;
    for (int j=0; j<mH*mW; j++)
    {
	  float e = lvis0[j]-lvisN[j];
	  v += e*e;
    }

    error[0] = v;
}








__kernel void clGetMap(__write_only image2d_t image, __global float* lhid)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int w = get_global_size(0);

    float l = lhid[y*w+x];
    write_imagef(image, (int2)(x,y), (float4)(l,l,l,1)); 
}

__kernel void clGetVis(__write_only image2d_t image, __global float* lvis)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int w = get_global_size(0);

    float l = lvis[y*w+x];
    write_imagef(image, (int2)(x,y), (float4)(l,l,l,1)); 
}

__kernel void clGetImage(__write_only image2d_t image, __global float* lvis)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int w = get_global_size(0);

    float l = lvis[y*w+x];
    write_imagef(image, (int2)(x,y), (float4)(l,l,l,1)); 
}



__kernel void clGetWeight(__write_only image2d_t image, __global float* weight, int iDx, int iDy)
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int w = get_global_size(0);

    float v =  1/(1+exp(-weight[y*w+x]));
    write_imagef(image, (int2)(iDx+x,iDy+y), (float4)(v,v,v,1)); 
}
