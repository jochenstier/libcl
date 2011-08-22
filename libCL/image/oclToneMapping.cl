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

__kernel void clLuminance(__read_only image2d_t RGBAin, __write_only image2d_t LUMout)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;

    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    const int gw = get_global_size(0);

	float4 RGBA = read_imagef(RGBAin, sampler, (float2)(gx,gy));

	write_imagef(LUMout, (int2)(gx,gy), dot((float4)(0.2126,0.7152,0.0722,0.0),RGBA));
}

__kernel void clCombine(__read_only image2d_t RGBDin, __read_only image2d_t LUMin, __write_only image2d_t RGBDout)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP;

    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    const int gw = get_global_size(0);
    const int gh = get_global_size(1);

	float2 pixel = (float2)((gx+0.5)/gw,(gy+0.5)/gh);

	float4 RGBA = read_imagef(RGBDin, sampler, pixel);
	float4 LLLL = read_imagef(LUMin, sampler, pixel);

	float luminance = dot((float4)(0.2126,0.7152,0.0722,0.0),RGBA);


	float glare = 0.0;//log(l) * (1.0 - 0.99/(0.99 + log(l)));

	// Retinex
	float4 result = RGBA*(exp(glare + log(luminance)-0.45f*log(LLLL)));

	// Ashikhmin
	//float4 result = RGBA*(exp(glare + log(luminance)-0.45*log(l)));

	//write_imagef(RGBDout, (int2)(gx,gy), Lum[gy*gw+gx]);
	write_imagef(RGBDout, (int2)(gx,gy), result);
}

