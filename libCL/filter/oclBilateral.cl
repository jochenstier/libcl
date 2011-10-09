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

__kernel void clBilateral(__read_only image2d_t imageIn, __write_only image2d_t imageOut,  int r, float4 scalar, int w, int h)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_REPEAT;
    int x = mul24(get_group_id(0), get_local_size(0)) + get_local_id(0);
    int y = mul24(get_group_id(1), get_local_size(1)) + get_local_id(1);

    if (x < w && y < h) 
	{
        float4 sum = 0.0f;
        float4 t = 0.0f;
        float4 factor;

        float4 dc = read_imagef(imageIn, sampler, (float2)(x,y)); 
		for(int i = -r; i <= r; i++)
		{
			for(int j = -r; j <= r; j++)
			{
				int lx = min(max(x+j, 0), w-1); 
				int ly = min(max(y+i, 0), h-1);
				float4 dp = read_imagef(imageIn, sampler, (float2)(lx,ly)); 

				// range domain
				float r2 = 0.5f*dot(dc-dp,scalar*scalar);
				float g = exp(-r2*r2);

				// spatial domain
				float4 r = dp.w;
				float4 w = exp(-r*r);

				t += dp * w * g;
				sum += w * g;
			}
		}

		write_imagef(imageOut, (int2)(x,y), (float4)(t/sum));
	}
}
