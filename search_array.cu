/*
CUDA search binary tree and bilinear interpolation in the triangle v1.0:

Copyright (C) 2010 Brown University

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301
USA, or see <http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>.

Author: Zhu Xueyu
*/



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>

#define CUDA_CHK(NAME, ARGS) { \
  cudaError_t cuda_err_code = NAME ARGS; \
  if (cuda_err_code != cudaSuccess) { \
    printf("%s failed with code %d\n", #NAME, cuda_err_code); \
    abort(); \
  } \
}

// using texture memory
texture<float, 1, cudaReadModeElementType> texRef;
texture<float, 1, cudaReadModeElementType> texRef2;
texture<float, 1, cudaReadModeElementType> valx_t;
texture<float, 1, cudaReadModeElementType> valy_t;
//const int max_threads = 512;
// input generation

#define MULT 1103515245
#define ADD 12345
#define MASK 0x7FFFFFFF
#define TWOTO31 2147483648.0

static int A = 1;
static int B = 0;
static int randx = 1;
static int lastrand;


static void drndset(int seed)
{
   A = 1;
   B = 0;
   randx = (A * seed + B) & MASK;
   A = (MULT * A) & MASK;
   B = (MULT * B + ADD) & MASK;
}


static double drnd()
{
   lastrand = randx;
   randx = (A * randx + B) & MASK;
   return (double)lastrand / TWOTO31;
}
    
//__constant__ int* x_ming, *x_maxg, *y_ming, *y_maxg;

__global__ void search_kernel(int len,int N, float *value_x, float* value_y, int *index, int *level_list_d, int* leaf_list_d, float* centerx_list_d, float *centery_list_d){
    
    int i;
    float xmin, ymin, xmax, ymax, width; 
    i = threadIdx.x + blockIdx.x * blockDim.x;
if (i<len){
    width = powf(2.0,-level_list_d[i]);
    xmin = centerx_list_d[i] - width;
    ymin = centery_list_d[i] - width;
    xmax = centerx_list_d[i] + width;
    ymax = centery_list_d[i] + width;
}
    const int s_width=512;
    __shared__ float x_loc[s_width], y_loc[s_width];
      int bound = max(N,len);
      for (int m = 0;m<N/s_width +1 ;m++)
      { 
        int k = m*s_width + threadIdx.x;
        if(k<N && threadIdx.x<s_width){
         x_loc[threadIdx.x] = value_x[m*s_width + threadIdx.x];
         y_loc[threadIdx.x] = value_y[m*s_width + threadIdx.x];
        }
        __syncthreads();  
 
        if (i< len){
           for (int j=0;j<s_width ; j++){
           if (x_loc[j] >= xmin && x_loc[j]<=xmax &&
               y_loc[j] > ymin && y_loc[j]<=ymax )
              index[j+m*s_width] = i;    
             //index[j] = leaf_list_d[i];    
            }

        }
        __syncthreads();   
      }
}


void search_cpu(int len,int N, float *value_x, float* value_y, int *index_cpu, int *level_list, int* leaf_list, float* centerx_list, float *centery_list){
       for (int i = 0; i< len; i++){
           float width = pow(2.0,-level_list[i]);   
           float xmin = centerx_list[i] - width;
           float ymin = centery_list[i] - width;
           float xmax = centerx_list[i] + width;
           float ymax = centery_list[i] + width;
          for (int j=0;j<N;j++){
           if (value_x[j] >= xmin && value_x[j]<=xmax &&
            value_y[j]>ymin && value_y[j]<=ymax)
              //index_cpu[j] = leaf_list[i];
              index_cpu[j] = i ;
          }
       }
}
    
#if 1
__global__ void interpolation(int N, float* value_x, float *value_y, int* index_g, int *level_list_d, float *centerx_list_d, float* centery_list_d,  float *T1_list_d, float* T2_list_d, float * T3_list_d, float* T4_list_d,float* interp_value){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i< N){
	    int j = index_g[i];
	    float width = powf(2.0,-level_list_d[j]);
#if 1
	    float xmin = centerx_list_d[j] - width;
	    float ymin = centery_list_d[j] - width;
	    float xmax = centerx_list_d[j] + width;
	    float ymax = centery_list_d[j] + width; 

#endif
#if 0
    float xmin =tex1Dfetch(texRef, i)  - width;
    float ymin =tex1Dfetch(texRef2, i)  - width;
    float xmax = tex1Dfetch(texRef, i) + width;
    float ymax = tex1Dfetch(texRef2, i) + width;
#endif
	    // rescale x,y in the local cell
	    float x_ref = (value_x[i]-xmin)/(xmax-xmin);
	    float y_ref = (value_y[i]-ymin)/(ymax-xmin);
	   
	    // pickup the interpolation triangle 
            float x_nodes[3], y_nodes[3], var[3];
	    x_nodes[0] = xmin;
	    x_nodes[1] = xmax ;
	    x_nodes[2] = x_ref>=y_ref?  xmax: xmin;
	    
            y_nodes[0] = ymin;
	    y_nodes[1] = x_ref>=y_ref? ymin:ymax ;
	    y_nodes[2] = ymax ;
	    
            var[0] = T1_list_d[j];
	    var[1] = x_ref>=y_ref? T2_list_d[j]: T3_list_d[j] ;
	    var[2] = x_ref>=y_ref? T3_list_d[j]: T4_list_d[j];
	float A = y_nodes[0]*(var[1]- var[2])  
                  +  y_nodes[1]*(var[2] - var[0]) 
                  +  y_nodes[2]*(var[0] - var[1]);

	float B = var[0]*(x_nodes[1] - x_nodes[2])
                   + var[1]*(x_nodes[2] - x_nodes[0])
                   +  var[2]*(x_nodes[0] - x_nodes[1]);

	float C = x_nodes[0]*(y_nodes[1] - y_nodes[2])
                  + x_nodes[1]*(y_nodes[2] - y_nodes[0])
                  + x_nodes[2]*(y_nodes[0] - y_nodes[1]);

	float D = -A*x_nodes[0] - B*y_nodes[0] - C*var[0];
	interp_value[i] = -(A*value_x[i] + B*value_y[i] + D)/C;
   }
}
#endif

void interpolation_cpu(int N, float* value_x, float *value_y, int* index_g, int *level_list_d, float *centerx_list_d, float* centery_list_d,  float *T1_list_d, float* T2_list_d, float * T3_list_d, float* T4_list_d,float* interp_value){
    std::cout << "interpolation on cpu!"<<std::endl;
    for(int i = 0;i< N;i++){
	    int j = index_g[i];
	    float width = powf(2.0,-level_list_d[j]);
	    float xmin = centerx_list_d[j] - width;
	    float ymin = centery_list_d[j] - width;
	    float xmax = centerx_list_d[j] + width;
	    float ymax = centery_list_d[j] + width; 

	    // rescale x,y in the local cell
	    float x_ref = (value_x[i]-xmin)/(xmax-xmin);
	    float y_ref = (value_y[i]-ymin)/(ymax-xmin);
	   
	    // pickup the interpolation triangle 
	    float x_nodes[3], y_nodes[3], var[3];
	    x_nodes[0] = xmin;
	    x_nodes[1] = x_ref>=y_ref?  xmax: xmax ;
	    x_nodes[2] = x_ref>=y_ref?  xmax: xmin;

	    y_nodes[0] = ymin;
	    y_nodes[1] = x_ref>=y_ref? ymin:ymax ;
	    y_nodes[2] = x_ref>=y_ref? ymax:ymax ;
	   
	    var[0] = T1_list_d[j];
	    var[1] = x_ref>=y_ref? T2_list_d[j]: T3_list_d[j] ;
	    var[2] = x_ref>=y_ref? T3_list_d[j]: T4_list_d[j];

	float A = y_nodes[0]*(var[1]- var[2])  +  y_nodes[1]*(var[2] - var[0]) +  y_nodes[2]*(var[0] - var[1]);

	float B = var[0]*(x_nodes[1] - x_nodes[2]) + var[1]*(x_nodes[2] - x_nodes[0]) +  var[2]*(x_nodes[0] - x_nodes[1]);

	float C = x_nodes[0]*(y_nodes[1] - y_nodes[2]) + x_nodes[1]*(y_nodes[2] - y_nodes[0]) + x_nodes[2]*(y_nodes[0] - y_nodes[1]);

	float D = -A*x_nodes[0] - B*y_nodes[0] - C*var[0];
	interp_value[i] = -(A*value_x[i] + B*value_y[i] + D)/C;

   }
}

int main( int argc, char** argv)
{
    // ----------------------v-------------------------------------------------
    CUDA_CHK(cudaSetDevice, (3)); // EDIT ME!
    // ----------------------^-------------------------------------------------
   
        // Read the database
        using namespace std;
	int num_nodes, num_leafs;
	float rootwidth, xmin, xmax, ymin, ymax;
	int *level_list, *leaf_list;
        float *centerx_list, *centery_list;
	float *T1_list, *T2_list, *T3_list, *T4_list,*P1_list, *P2_list
	,*P3_list, *P4_list; // variable lists
        
        ifstream myfile("RPTBDB.dat");
        myfile >> num_nodes;
        myfile >> ymin >> ymax >> xmin >> xmax;
        myfile >> num_leafs >> rootwidth >> rootwidth;
        
        unsigned int bytes; 
        int fbytes = num_leafs*sizeof(float);
 	bytes = num_leafs * sizeof(int);
	int dbytes = sizeof(float);
	
        level_list = (int *) malloc( bytes);
	leaf_list = (int  *) malloc( bytes);
	centerx_list = (float *) malloc( fbytes);
	centery_list= (float *) malloc( fbytes);
	T1_list = (float *) malloc( fbytes);
	T2_list = (float *) malloc( fbytes);
	T3_list= (float *) malloc( fbytes);
	T4_list= (float *) malloc( fbytes);
	P1_list = (float *) malloc( fbytes);
	P2_list = (float *) malloc( fbytes);
	P3_list= (float *) malloc( fbytes);
	P4_list= (float *) malloc( fbytes);
        if (myfile.is_open())
	{
	  for(int i=0;i< num_leafs; i++){
	     myfile >> level_list[i] >> leaf_list[i];
	     myfile >> centerx_list[i] >> centery_list[i];
	     myfile >> T1_list[i] >> P1_list[i];
	     myfile >> T2_list[i] >> P2_list[i];
	     myfile >> T3_list[i] >> P3_list[i];
	     myfile >> T4_list[i] >> P4_list[i];
             }
	}
	myfile.close();

	int size= num_leafs; // numbet of elements to reduce 

	// allocate variables on GPU
	int *level_list_d, *leaf_list_d;
        float *centerx_list_d, *centery_list_d;
	float *T1_list_d, *T2_list_d, *T3_list_d, *T4_list_d,*P1_list_d, *P2_list_d;
        int *index, *index_g =NULL;
        float *value_x, *value_y, *value_x_d, *value_y_d, *interp_h, *interp_d, *interp;
       
        //rescale the input the data 
        int N=100*1000;
        value_x = (float *) malloc( N*dbytes);
        value_y = (float *) malloc( N*dbytes);
        index = (int *) malloc( N*sizeof(int));
        interp_h = (float *) malloc( N*dbytes);
        interp = (float *) malloc( N*dbytes);

        drndset(9);
        int *index_cpu;
        index_cpu = (int *) malloc( N*sizeof(int));

        for (int i=0; i < N; i++){
		value_x[i] = drnd()*600 + 400;
		value_y[i] = drnd()*2.0 - 1.0;
		value_x[i] = (value_x[i]-xmin)/(xmax-xmin);
		value_y[i] = (value_y[i]-ymin)/(ymax-ymin);
                index[i] = -1;
                index_cpu[i]=-1;
                interp[i] = -1;
                interp_h[i] = -1;
                //cout << i << " " <<value_x[i] << " " << value_y[i]<<endl;
        }
    
    clock_t starttime, endtime; 
    starttime = clock();
    search_cpu(size, N, value_x, value_y, index_cpu, level_list, leaf_list, centerx_list, centery_list);
    interpolation_cpu(N, value_x, value_y, index_cpu, level_list,centerx_list, centery_list, T1_list, T2_list, T3_list, T4_list,interp);
    endtime = clock();
    
    // allocate device memory and data
    cout << "allocating memory on GPU!" << endl;
    CUDA_CHK(cudaMalloc, ((void**) &level_list_d, size*sizeof(int)));
    CUDA_CHK(cudaMalloc, ((void**) &leaf_list_d, size*sizeof(int)));
    CUDA_CHK(cudaMalloc, ((void**) &centerx_list_d, size*sizeof(float)));
    CUDA_CHK(cudaMalloc, ((void**) &centery_list_d, size*sizeof(float)));
    
    CUDA_CHK(cudaMalloc, ((void**) &value_x_d, N*sizeof(float)));
    CUDA_CHK(cudaMalloc, ((void**) &value_y_d, N*sizeof(float)));
    CUDA_CHK(cudaMalloc, ((void**) &index_g, N*sizeof(int)));
#if 1    
    CUDA_CHK(cudaBindTexture, (0,texRef,centerx_list_d, size*sizeof(float)));
    CUDA_CHK(cudaBindTexture, (0,texRef2,centery_list_d, size*sizeof(float)));
    CUDA_CHK(cudaBindTexture, (0,valx_t,value_x_d, N*sizeof(float)));
    CUDA_CHK(cudaBindTexture, (0,valy_t,value_y_d, N*sizeof(float)));
#endif
#if 1 
    CUDA_CHK(cudaMalloc, ((void**) &T1_list_d, fbytes));
    CUDA_CHK(cudaMalloc, ((void**) &T2_list_d, fbytes));
    CUDA_CHK(cudaMalloc, ((void**) &T3_list_d, fbytes));
    CUDA_CHK(cudaMalloc, ((void**) &T4_list_d, fbytes));
    CUDA_CHK(cudaMalloc, ((void**) &interp_d, N*dbytes));
#endif
    cout << "transfering data to GPU!" << endl;
    CUDA_CHK(cudaMemcpy, (level_list_d,level_list, size*sizeof(int),
                cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy, (leaf_list_d,leaf_list, size*sizeof(int),
                cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy, (centerx_list_d,centerx_list, size*sizeof(float),
                cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy, (centery_list_d,centery_list, size*sizeof(float),
                cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy, (value_x_d,value_x,N*sizeof(float),
                cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy, (value_y_d,value_y,N*sizeof(float),
                cudaMemcpyHostToDevice));
 #if 1  
    CUDA_CHK(cudaMemcpy, (T1_list_d, T1_list, fbytes,
                cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy, (T2_list_d, T2_list, fbytes,
                cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy, (T3_list_d, T3_list, fbytes,
               cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy, (T4_list_d, T4_list, fbytes,
                cudaMemcpyHostToDevice));
    
    CUDA_CHK(cudaMemcpy, (interp_d,interp_h,N*dbytes,
                cudaMemcpyHostToDevice));
#endif
    CUDA_CHK(cudaMemcpy, (index_g, index, N*sizeof(int),
                cudaMemcpyHostToDevice));

    cout << "launching the kernel..." << endl;
    // run the kernel
    int num_threads = 512;
    int num_blocks = (size+ num_threads - 1)/num_threads ;
    cout << size << " "<<  num_threads << " " << num_blocks << " here" << endl;
    // measure the time
    
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);  cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // locate the cell
    search_kernel<<<num_blocks, num_threads>>>(size, N, value_x_d, value_y_d, index_g, level_list_d, leaf_list_d, centerx_list_d, centery_list_d); 

    printf("search kernel finished!\n");
    cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
    float search_time = time;
    num_blocks = (N+ num_threads - 1)/num_threads ;
    cout << size << " "<<  num_threads << " " << num_blocks << " here" << endl;
    interpolation<<<num_blocks, num_threads>>>(N, value_x_d, value_y_d, index_g, level_list_d,centerx_list_d, centery_list_d, T1_list_d, T2_list_d, T3_list_d, T4_list_d,interp_d);
   
    cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);

    float interpolation_time = time-search_time;
    cudaEventDestroy(start);  cudaEventDestroy(stop);

    printf("interpolation kernel finished!\n");
    
    // copy back the result
    CUDA_CHK(cudaMemcpy, (index,index_g, N*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHK(cudaMemcpy, (interp_h, interp_d, N*sizeof(float), cudaMemcpyDeviceToHost));

    // check the result
    for (int i=0; i < N; i++){
       assert(interp[i]=interp_h[i]);
    if (index[i]<0)
       printf("cell %d is not in this range!, cpu: %d\n", i, index_cpu[i]);
    
    //else
       //printf("the value is cell : %d %d \n",index[i],index_cpu[i] ); 
    //   printf("the value is cell : %d %d %f %f\n",index[i],index_cpu[i], interp_h[i], interp[i] ); 
    }
  
    //output the time
     //printf("GPU: %.1f ms\n", time);
     printf("GPU: search_time: %10.5f ms, interpolation_time: %10.5f ms, total_time: %10.5f ms\n", search_time, interpolation_time, search_time + interpolation_time);
     printf("CPU %ld ms\n", (int) (1000.0f * (endtime - starttime) / CLOCKS_PER_SEC));

  
    //clean up
  CUDA_CHK(cudaUnbindTexture,(texRef));
  CUDA_CHK(cudaUnbindTexture,(texRef2));

	CUDA_CHK(cudaFree, (level_list_d);  );
	CUDA_CHK(cudaFree, (leaf_list_d);   );
	CUDA_CHK(cudaFree, (centerx_list_d););
	CUDA_CHK(cudaFree, (centery_list_d););
	CUDA_CHK(cudaFree, (value_x_d));
	CUDA_CHK(cudaFree, (value_y_d));
	CUDA_CHK(cudaFree, (index_g));
	
        free(level_list);
	free(leaf_list);
	free(centerx_list);
	free(centery_list);
	free(value_x);
	free(value_y);
}
