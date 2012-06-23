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
#ifndef _oclColor
#define _oclColor

#include "oclProgram.h"
#include "oclImage2D.h"

class oclColor : public oclProgram
{
    public: 

        oclColor(oclContext& iContext);

        int compile();
        
        // apply separable convolution over iAxis to 3D buffer
        int RGBtoHSV(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest)
        {
            return invoke(iDevice, bfSrce, bfDest, clRGBtoHSV);
        };
        int HSVtoRGB(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest)
        {
            return invoke(iDevice, bfSrce, bfDest, clHSVtoRGB);
        };


        int RGBtoXYZ(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest)
        {
            return invoke(iDevice, bfSrce, bfDest, clRGBtoXYZ);
        };
        int XYZtoRGB(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest)
        {
            return invoke(iDevice, bfSrce, bfDest, clXYZtoRGB);
        };

        int RGBtoLAB(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest)
        {
            return invoke(iDevice, bfSrce, bfDest, clRGBtoLAB);
        };
        int LABtoRGB(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest)
        {
            return invoke(iDevice, bfSrce, bfDest, clLABtoRGB);
        };


    protected:

        int invoke(oclDevice& iDevice, oclImage2D& bfSrce, oclImage2D& bfDest, oclKernel& iKernel);

        oclKernel clHSVtoRGB;
        oclKernel clRGBtoHSV;

        oclKernel clRGBtoXYZ;
        oclKernel clXYZtoRGB;

        oclKernel clRGBtoLAB;
        oclKernel clLABtoRGB;
};      

#endif