#include "stdio.h"
#include <climits>
#include <cstdlib>
#include <endian.h>
#include <sys/wait.h>

static void HandleError(cudaError_t err,
    const char *file,int line){
  if (err!=cudaSuccess){
    printf("%s in %s at line %d\n",cudaGetErrorString(err),file,line);
    exit(EXIT_FAILURE);
  }
}

#define CHECK(error){HandleError(error,__FILE__,__LINE__);}

int getThreadNum(){
  cudaDeviceProp prop;
  int count;

  CHECK(cudaGetDeviceCount(&count));
  printf("gpu num %d\n",count);
  CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("max thread num: %d\n",prop.maxThreadsPerBlock);
  printf("max grid dimemsions: %d, %d,%d)\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
  return prop.maxThreadsPerBlock;
}


__global__ void conv(float *img,float *kernel,float *result,const int width,const int height,const int kernel_size){
  int ti=threadIdx.x;
  int bi=blockIdx.x;
  int id=(bi*blockDim.x+ti);
  if (id>=width*height){
    return;
  }

  int row=id/width;
  int col=id%width;

  for (int i=0;i<kernel_size;++i){
    for (int j=0;j<kernel_size;++j){
      float img_value=0;
      int curRow=row-kernel_size/2+i;
      int curCol=col-kernel_size/2+j;
      if (curRow<0||curCol<0||curRow>=height||curCol>=width){
      }else{
        img_value=img[curRow*width+curCol];
      }
      result[id]+=kernel[i*kernel_size+j]*img_value;
    }
  }
}

int main(void){
  const int width=1920;
  const int height=1080;
  float *img=new float[width*height];
  float *result=new float[width*height];
  for (int row=0;row<height;++row){
    for (int col=0;col<width;++col){
      img[col+row*width]=(col+row)%256;
    }
  }

  const int kernel_size=3;
  float *kernel=new float[kernel_size*kernel_size];
  for (int i=0;i<kernel_size*kernel_size;++i){
    kernel[i]=i%kernel_size-1;
  }
  //visualization
  
  for (int row=0;row<10;++row){
    for (int col=0;col<10;++col){
      printf("%2.0f ",img[col+row*width]);
    }
    printf("\n");
  }
  printf("kernel\n");
  for (int row=0;row<kernel_size;++row){
    for (int col=0;col<kernel_size;++col){
      printf("%2.0f ",kernel[col+row*kernel_size]);
    }
    printf("\n");
  }

  float *img_gpu;
  float *kernel_gpu;
  float *result_gpu;

  CHECK(cudaMalloc((void**)&img_gpu,width*height*sizeof(float)));
  CHECK(cudaMalloc((void**)&kernel_gpu,kernel_size*kernel_size*sizeof(float)));
  CHECK(cudaMalloc((void**)&result_gpu,width*height*sizeof(float)));
  CHECK(cudaMemcpy(img_gpu,img,width*height*sizeof(float),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(kernel_gpu,kernel,kernel_size*kernel_size*sizeof(float),cudaMemcpyHostToDevice));

  int threadNum=getThreadNum();
  int blockNum=(width*height-0.5)/threadNum+1;

  conv<<<blockNum,threadNum >>>(img_gpu,kernel_gpu,result_gpu,width,height,kernel_size);
  CHECK(cudaMemcpy(result,result_gpu,width*height*sizeof(float),cudaMemcpyDeviceToHost));

  printf("result\n");
  for (int row=0;row<10;++row){
    for (int col=0;col<10;++col){
      printf("%2.0f ",result[col+row*width]);
    }
    printf("\n");
  }
  return 0;
}
