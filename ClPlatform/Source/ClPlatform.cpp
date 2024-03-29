#include "ClPlatform.hpp"
#include "stl.hpp"

ClPlatform& ClPlatform::getPlatform()
{
  static ClPlatform onlyPlatform;
  return onlyPlatform;
}

ClPlatform::ClPlatform()
{
  cl_uint platformNb;
  cl_int error = clGetPlatformIDs( 1, &platform, &platformNb );
  if( (error != CL_SUCCESS) || (platformNb < 1) )
  {
      std::cerr << "clGetPlatformIDs OpenCL error = " << error << std::endl;
      setUpSuccessfully=false;
      return;
  }

  cl_uint deviceNb;
  error = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, &deviceNb );
  if( (error != CL_SUCCESS) || (deviceNb < 1) || (!device) )
  {
      std::cerr  << "clGetDeviceIDs OpenCL error = " << error << std::endl;
      setUpSuccessfully=false;
      return;
  }

  context = clCreateContext(0, 1, &device, NULL, NULL, &error);
  if( (error != CL_SUCCESS) || (!context) )
  {
      std::cerr << "clCreateContext OpenCL error = " << error  << std::endl;
      setUpSuccessfully=false;
      return;
  }

  queue = clCreateCommandQueue(context, device, 0, &error);
  if( (error != CL_SUCCESS) || (!queue) )
  {
      std::cerr << "clCreateCommandQueue OpenCL error = " << error  << std::endl;
      setUpSuccessfully=false;
      return;
  }
  
  error = clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(size_t),&max_work_group_size,NULL); // TODO create class DeviceInfo and move it there
  if(error != CL_SUCCESS)
  {
      std::cerr << "clGetDeviceInfo OpenCL error = " << error  << std::endl;
      setUpSuccessfully=false;
      return;
  }

  bool printDeviceInfo = false;
  if(printDeviceInfo)
  {
      char device_name[100];
      clGetDeviceInfo(device,CL_DEVICE_NAME,100,device_name,NULL); 
      std::cout << "Device name: " << device_name << std::endl;

	
      cl_device_local_mem_type local_mem_type;
      clGetDeviceInfo(device,CL_DEVICE_LOCAL_MEM_TYPE	 ,sizeof(cl_device_local_mem_type),&local_mem_type,NULL); 
      switch(local_mem_type)
      {
        case CL_LOCAL:
          std::cout << "Local mem type is CL_LOCAL" << std::endl;
          break;
        case CL_GLOBAL:
          std::cout << "Local mem type is CL_GLOBAL" << std::endl;
          break;
      }

      cl_ulong local_mem_size=0;
      clGetDeviceInfo(device,CL_DEVICE_LOCAL_MEM_SIZE	 ,sizeof(cl_ulong),&local_mem_size,NULL); 
      std::cout << "Local mem size = " << local_mem_size << " bytes = " << (local_mem_size/1024) << " kB" << std::endl;

      cl_uint max_comp_units=0;
      clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS ,sizeof(cl_uint),&max_comp_units,NULL);
      std::cout << "Max conpute units = " << max_comp_units << std::endl;

      cl_uint pref_vect_width_float=0;
      clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT ,sizeof(cl_uint),&pref_vect_width_float,NULL);
      std::cout << "Prefered float vector width = " << pref_vect_width_float << std::endl;


  }

  setUpSuccessfully = true;
}

bool ClPlatform::isSetUpSuccessfully()
{
	return setUpSuccessfully;
}

void ClPlatform::execute() const
{
  cl_int error = clFinish(queue);
  if ( error != CL_SUCCESS )
  {
      std::cerr << "clFinish error = " << error << std::endl;
  }
}

cl_context ClPlatform::getContext() const
{
  return context;
}

cl_device_id ClPlatform::getDevice() const
{
  return device;
}

ClPlatform::~ClPlatform()
{
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}

