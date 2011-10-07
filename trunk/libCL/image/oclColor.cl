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


float minComp(float4 v)
{
    float t = (v.x<v.y) ? v.x : v.y;
    t = (t<v.z) ? t : v.z;
    return t;
}

float maxComp(float4 v)
{
    float t = (v.x>v.y) ? v.x : v.y;
    t = (t>v.z) ? t : v.z;
    return t;
}

//
// RGB - HSV
//

float4 RGBtoHSV(float4 RGB)
{
	float4 HSV = (float4)0;
	float minVal = minComp(RGB);
	float maxVal = maxComp(RGB);
	float delta = maxVal - minVal;           
	if (delta != 0) 
	{                   
		float4 delRGB = ((((float4)(maxVal) - RGB)/6.0) + (delta/2.0))/delta;

		HSV.y = delta / maxVal;

		if ( RGB.x == maxVal ) 
			HSV.x = delRGB.z - delRGB.y;
		else if (RGB.y == maxVal) 
			HSV.x = (1.0/3.0) + delRGB.x - delRGB.z;
		else if (RGB.z == maxVal) 
			HSV.x = (2.0/3.0) + delRGB.y - delRGB.x;

		if (HSV.x < 0.0)  
			HSV.x += 1.0;
		if (HSV.x > 1.0)
			HSV.x -= 1.0;
    }
	HSV.z = maxVal;
 	HSV.w = RGB.w;
    return (HSV);
}

float4 HSVtoRGB(float4 HSV)
{
	float4 RGB = (float4)0;
	if (HSV.z != 0) 
	{
		float var_h = HSV.x * 6;
		float var_i = floor(var_h-0.000001);   
		float var_1 = HSV.z * (1.0 - HSV.y);
		float var_2 = HSV.z * (1.0 - HSV.y * (var_h-var_i));
		float var_3 = HSV.z * (1.0 - HSV.y * (1-(var_h-var_i)));
		switch((int)(var_i))
		{
			case 0: RGB = (float4)(HSV.z, var_3, var_1, HSV.w); break;
			case 1: RGB = (float4)(var_2, HSV.z, var_1, HSV.w); break;
			case 2: RGB = (float4)(var_1, HSV.z, var_3, HSV.w); break;
			case 3: RGB = (float4)(var_1, var_2, HSV.z, HSV.w); break;
			case 4: RGB = (float4)(HSV.z, var_1, var_2, HSV.w); break;
			default: RGB = (float4)(HSV.z, var_1, var_2, HSV.w); break;
		}
	}
	RGB.w = HSV.w;
	return (RGB);
}

__kernel void clRGBtoHSV(__read_only image2d_t srce, __write_only image2d_t dest)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;
	int x = get_global_id(0);
	int y = get_global_id(1);
	int2 idx = (int2)(x,y);
	write_imagef(dest, idx, RGBtoHSV(read_imagef(srce, sampler, idx)));
}

__kernel void clHSVtoRGB(__read_only image2d_t srce, __write_only image2d_t dest)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;
	int x = get_global_id(0);
	int y = get_global_id(1);
	int2 idx = (int2)(x,y);
	write_imagef(dest, idx, HSVtoRGB(read_imagef(srce, sampler, idx)));
}


//
// RGB - CIE XYZ
//

float4 RGBtoXYZ(float4 RGB)
{
    float r = (RGB.x > 0.04045)? pow((RGB.x + 0.055)/(1 + 0.055), 2.2) : (RGB.x/12.92) ;
    float g = (RGB.y > 0.04045)? pow((RGB.y + 0.055)/(1 + 0.055), 2.2) : (RGB.y/12.92) ;
    float b = (RGB.z > 0.04045)? pow((RGB.z + 0.055)/(1 + 0.055), 2.2) : (RGB.z/12.92) ;

	return (float4)((r*0.4124 + g*0.3576 + b*0.1805),
					(r*0.2126 + g*0.7152 + b*0.0722),
					(r*0.0193 + g*0.1192 + b*0.9505),
					RGB.w
				  );
}

float4 XYZtoRGB(float4 XYZ)
{
    float4 val;
    val.x =  XYZ.x*3.2410 - XYZ.y*1.5374 - XYZ.z*0.4986;
    val.y = -XYZ.x*0.9692 + XYZ.y*1.8760 - XYZ.z*0.0416;
    val.z =  XYZ.x*0.0556 - XYZ.y*0.2040 + XYZ.z*1.0570;

	val.x = (val.x<=0.0031308)? 12.92*val.x : (1+0.055)* pow(val.x, (1.0/2.4)) - 0.055;
	val.y = (val.y<=0.0031308)? 12.92*val.y : (1+0.055)* pow(val.y, (1.0/2.4)) - 0.055;
	val.z = (val.z<=0.0031308)? 12.92*val.z : (1+0.055)* pow(val.z, (1.0/2.4)) - 0.055;

	val.w = XYZ.w;
    return val;
}

__kernel void clRGBtoXYZ(__read_only image2d_t srce, __write_only image2d_t dest)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;
	int x = get_global_id(0);
	int y = get_global_id(1);
	int2 idx = (int2)(x,y);
	write_imagef(dest, idx, RGBtoXYZ(read_imagef(srce, sampler, idx)));
}

__kernel void clXYZtoRGB(__read_only image2d_t srce, __write_only image2d_t dest)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;
	int x = get_global_id(0);
	int y = get_global_id(1);
	int2 idx = (int2)(x,y);
	write_imagef(dest, idx, XYZtoRGB(read_imagef(srce, sampler, idx)));
}

//
// RGB - CIE LAB
//


float Ffor(float t)
{
    return ((t > 0.008856)? pow(t, (1.0/3.0)) : (7.787*t + 16.0/116.0));
}

float4 RGBtoLAB(float4 RGB)
{
	const float4 D65 = (float4)(0.9505, 1.0, 1.0890, 0.0);

	float4 XYZ = RGBtoXYZ(RGB);

	float4 LAB;
    LAB.x = 116.0 *  Ffor(XYZ.y/D65.y) -16;
    LAB.y = 500.0 * (Ffor(XYZ.x/D65.x) - Ffor(XYZ.y/D65.y));
    LAB.z = 200.0 * (Ffor(XYZ.y/D65.y) - Ffor(XYZ.z/D65.z));
	LAB.w = XYZ.w;
    return LAB;
}

float4 LABtoRGB(float4 LAB)
{
	const float4 D65 = (float4)(0.9505, 1.0, 1.0890, 0.0);

    float fy = (LAB.x+16)/116.0;
    float fx = fy + (LAB.y/500.0);
    float fz = fy - (LAB.z/200.0);

    float delta = 6.0/29.0;
    float4 XYZ = (float4)(
				(fx > delta)? D65.x * (fx*fx*fx) : (fx - 16.0/116.0)*3*(delta*delta)*D65.x,
				(fy > delta)? D65.y * (fy*fy*fy) : (fy - 16.0/116.0)*3*(delta*delta)*D65.y,
				(fz > delta)? D65.z * (fz*fz*fz) : (fz - 16.0/116.0)*3*(delta*delta)*D65.z,
				 LAB.w);

	return XYZtoRGB(XYZ);
}

__kernel void clRGBtoLAB(__read_only image2d_t srce, __write_only image2d_t dest)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;
	int x = get_global_id(0);
	int y = get_global_id(1);
	int2 idx = (int2)(x,y);
	write_imagef(dest, idx, RGBtoLAB(read_imagef(srce, sampler, idx)));
}

__kernel void clLABtoRGB(__read_only image2d_t srce, __write_only image2d_t dest)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;
	int x = get_global_id(0);
	int y = get_global_id(1);
	int2 idx = (int2)(x,y);
	write_imagef(dest, idx, LABtoRGB(read_imagef(srce, sampler, idx)));
}



//
//
//

__kernel void clQuantizeLAB(__read_only image2d_t srce, __write_only image2d_t dest)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;
	int x = get_global_id(0);
	int y = get_global_id(1);
	float4 LAB = read_imagef(srce, sampler, (int2)(x,y));

	const float binL = 10.2;
	const float binA = 20.1;
	const float sharpness = 10.099;

	float bL = LAB.x - fmod(LAB.x, binL) + binL/2.0;
	float bA = LAB.y - fmod(LAB.y, binA) + binA/2.0;
	LAB.x = bL + binL/2.0f*tanh(sharpness*(LAB.x - bL));
	LAB.y = bA + binL/2.0f*tanh(sharpness*(LAB.y - bA));

	write_imagef(dest, (int2)(x,y), LAB);
}


