## About ##

**libCL** is an open-source library for high performance computing in OpenCL. Rather than a specific domain, libCL intends to encompass a wide range of parallel algorithms. The goal is to provide a growing repository of kernels and data structures for visual-centric computing ranging from fundamental primitives such as sorting, searching and algebra to advanced systems of algorithms for computational research and visualization.



## Overview ##

**libCL** provides a core set of C++ classes that encapsulate the OpenCL API and provide a simple error handling and logging infrastrucuture. Based upon this core functionality are implementations of specific algorithms categorized within different subdirectories. An example of how to extend libCL is found on the **[wiki](CodeSample.md)** page. Up to this point, libCL contains following classes/algorithms:

**_color\_**
  * oclColor
  * oclQuantize

**_filter\_**
  * oclConvolute
  * oclBilateralFilter
  * oclTangent
  * oclSoebel
  * oclRecursiveGaussian
  * oclBilateralGrid
  * oclBilinearPyramid

**_phys\_**
  * oclFluid3D ([video](http://www.youtube.com/watch?v=g1m95ICzKTY&autoplay=1))

**_sort\_**
  * oclRadixSort

**_geom\_**
  * oclBvhTrimesh ([video](http://www.youtube.com/watch?v=bvjOXl4KUiM&autoplay=1))

**_image\_**
  * oclAmbientOcclusion
  * oclToneMapping
  * oclBloom

**_math\_**
  * oclVector

**_util\_**
  * oclMemory




## Editor ##

**libCL** closely integrates with <a href='http://www.opencldev.com'>OpenCL Studio</a> via the <a href='http://www.libcl.org/libBind.html'>libBind</a>**library. All of the demo applications bundled with OpenCL Studio are built upon libCL.**


