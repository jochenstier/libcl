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


__kernel void clMemSetImage(__write_only image2d_t dest, float4 value)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    write_imagef(dest, (int2)(x,y), value);
}

__kernel void clMemSetBuffer_float4(__global float4* dest, float4 value)
{
    dest[get_global_id(0)] = value;
}

__kernel void clMemSetBuffer_float(__global float* dest, float value)
{
    dest[get_global_id(0)] = value;
}

__kernel void clMean(__global const float* srce, __global float* mean, int size, int count, __local float* total)
{
    int g = get_global_id(0);
    int w = get_global_size(0);

    total[g] = 0.0f;
    for (int i=0; i<count; i++)
    {
	   int index = i*w+g;
	   if (index < size)
	   {
		  total[g] += srce[index];
	   }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (g == 0)
    {
       float v = 0.0f;
       for (int i=0; i<w; i++)
       {
	     v += total[i];
       }
       mean[0] = v/size;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

/*
__kernel void clMean(__global const float* srce, __global float* mean, int size)
{
    float v = 0.0f;
    for (int j=0; j<size; j++)
    {
	  v += srce[j];
    }
    mean[0] = v/size;
}
*/


__kernel void clVariance(__global const float* srce, __global const float* mean,  __global float* variance, int size, int count, __local float* total)
{
    int g = get_global_id(0);
    int w = get_global_size(0);

    total[g] = 0.0f;
    for (int i=0; i<count; i++)
    {
	   int index = i*w+g;
	   if (index < size)
	   {
	       float v = srce[index] - mean[0];
		   total[g] += v*v;
	   }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (g == 0)
    {
       float v = 0.0f;
       for (int i=0; i<w; i++)
       {
	     v += total[i];
       }
       variance[0] = v/size;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

/*
__kernel void clVariance(__global const float* srce, __global const float* mean,  __global float* variance, int size)
{
    float v = 0;
    for (int j=0; j<size; j++)
    {
	  v += (srce[j] - mean[0])*(srce[j] - mean[0]);
    }
    variance[0] = v/size;
}
*/

__kernel void clNormalize(__global float* srce, __global float* mean, __global float* variance, __global float* dest)
{
    const int i = get_global_id(0);

    dest[i] = (srce[i] - mean[0])/sqrt(variance[0]);  // contrast and variance normalization
}



__kernel void clMin(__global const float* srce, __global float* result, int size)
{
    float v = FLT_MAX;
    for (int j=0; j<size; j++)
    {
		v = min(v, srce[j]);
    }
    result[0] = v;
}


__kernel void clMax(__global const float* srce, __global float* result, int size)
{
    float v = -FLT_MAX;
    for (int j=0; j<size; j++)
    {
		v = max(v, srce[j]);
    }
    result[0] = v;
}
