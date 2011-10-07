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

__kernel void clConvoluteBuffer3D(__global float4* srce, __global float4* dest, int4 axis, __constant float* filter, int size)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	int sx = get_global_size(0);
	int sy = get_global_size(1);
	int sz = get_global_size(2);

	int limit = size/2;
	float4 DEST = (float4)0;
	for (int i=-limit; i<limit+1; i++)
	{
		int gx = min(max(x+i*axis.x,0),sx-1);
		int gy = min(max(y+i*axis.y,0),sy-1);
		int gz = min(max(z+i*axis.z,0),sz-1);
		float4 SRCE = srce[gz*sx*sy+gy*sx+gx];
		DEST += SRCE*filter[i+limit];
	}
	dest[z*sx*sy+y*sx+x] = DEST;
}
