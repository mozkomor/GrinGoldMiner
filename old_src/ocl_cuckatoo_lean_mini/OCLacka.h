#pragma once

#include "CL\cl.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <tchar.h>
#include <memory.h>
#include <vector>
#include <math.h>

// support variables
static int devID = 0;
static const char * platName = "NVIDIA";

static cl_ulong deviceMem;
static cl_ulong deviceMaxAlloc;
static std::vector<char> deviceName;

/*****************************************************************************
* Copyright (c) 2013-2016 Intel Corporation
* All rights reserved.
*
* WARRANTY DISCLAIMER
*
* THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
* MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* Intel Corporation is the author of the Materials, and requests that all
* problem reports or change requests be submitted to it directly
*/

void LogInfo(const char* str, ...)
{
	if (str)
	{

	}
}

void LogError(const char* str, ...)
{
	if (str)
	{

	}
}

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// Macros for OpenCL versions
#define OPENCL_VERSION_1_2  1.2f
#define OPENCL_VERSION_2_0  2.0f


struct ocl_args_d_t
{
	ocl_args_d_t();
	~ocl_args_d_t();

	// Regular OpenCL objects:
	cl_context       context;           // hold the context handler
	cl_device_id     device;            // hold the selected device handler
	cl_device_id     devices[8];            // hold the selected device handler
	cl_command_queue commandQueue;      // hold the commands-queue handler
	cl_program       program;           // hold the program handler
	cl_kernel        kernel;            // hold the kernel handler
	cl_kernel        kernel2;            // hold the kernel handler
	cl_kernel        kernel3;            // hold the kernel handler
	cl_kernel        kernel4;            // hold the kernel handler
	cl_kernel        kernel5;            // hold the kernel handler
	cl_kernel        kernel6;            // hold the kernel handler
	cl_kernel        kernel7;            // hold the kernel handler
	cl_kernel        kernel8;            // hold the kernel handler
	cl_kernel        kernel9;            // hold the kernel handler
	float            platformVersion;   // hold the OpenCL platform version (default 1.2)
	float            deviceVersion;     // hold the OpenCL device version (default. 1.2)
	float            compilerVersion;   // hold the device OpenCL C version (default. 1.2)

										// Objects that are specific for algorithm implemented in this sample
	cl_mem           srcA;              // hold first source buffer
	cl_mem           srcB;              // hold second source buffer
	cl_mem           srcC;              // hold second source buffer
	cl_mem           srcD;              // hold second source buffer
	cl_mem           dstMem;            // hold destination buffer
	cl_mem           dstMemA;            // hold destination buffer
	cl_mem           dstMemB;            // hold destination buffer
};


ocl_args_d_t::ocl_args_d_t() :
	context(NULL),
	device(NULL),
	commandQueue(NULL),
	program(NULL),
	kernel(NULL),
	kernel2(NULL),
	kernel3(NULL),
	kernel4(NULL),
	kernel5(NULL),
	kernel6(NULL),
	kernel7(NULL),
	kernel8(NULL),
	kernel9(NULL),
	platformVersion(OPENCL_VERSION_1_2),
	deviceVersion(OPENCL_VERSION_1_2),
	compilerVersion(OPENCL_VERSION_1_2),
	srcA(NULL),
	srcB(NULL),
	srcC(NULL),
	srcD(NULL),
	dstMem(NULL),
	dstMemA(NULL),
	dstMemB(NULL)
{
}

const char* TranslateOpenCLError(cl_int errorCode)
{
	switch (errorCode)
	{
	case CL_SUCCESS:                            return "CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";                          //-13
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";   //-14
	case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";                               //-15
	case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";                                  //-16
	case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";                                  //-17
	case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";                               //-18
	case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";                         //-19
	case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
	case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
	case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
	case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";                           //-63
	case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";                                   //-64
	case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";                           //-65
	case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";                           //-66
	case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";                             //-67
	case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";                     //-68
																												//    case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";                                  //-69
																												//    case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";                               //-70    

	default:
		return "UNKNOWN ERROR CODE";
	}
}


/*
* destructor - called only once
* Release all OpenCL objects
* This is a regular sequence of calls to deallocate all created OpenCL resources in bootstrapOpenCL.
*
* You may want to call these deallocation procedures in the middle of your application execution
* (not at the end) if you don't further need OpenCL runtime.
* You may want to do that in order to free some memory, for example,
* or recreate OpenCL objects with different parameters.
*
*/
ocl_args_d_t::~ocl_args_d_t()
{
	cl_int err = CL_SUCCESS;

	if (kernel)
	{
		err = clReleaseKernel(kernel);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (kernel2)
	{
		err = clReleaseKernel(kernel2);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (kernel3)
	{
		err = clReleaseKernel(kernel3);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (kernel4)
	{
		err = clReleaseKernel(kernel4);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (kernel5)
	{
		err = clReleaseKernel(kernel5);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (kernel6)
	{
		err = clReleaseKernel(kernel6);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (kernel7)
	{
		err = clReleaseKernel(kernel7);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (kernel8)
	{
		err = clReleaseKernel(kernel8);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (kernel9)
	{
		err = clReleaseKernel(kernel9);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (program)
	{
		err = clReleaseProgram(program);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseProgram returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (srcA)
	{
		err = clReleaseMemObject(srcA);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (srcB)
	{
		err = clReleaseMemObject(srcB);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (srcC)
	{
		err = clReleaseMemObject(srcC);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (srcD)
	{
		err = clReleaseMemObject(srcD);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (dstMem)
	{
		err = clReleaseMemObject(dstMem);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (dstMemA)
	{
		err = clReleaseMemObject(dstMemA);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
		}
	}

	if (dstMemB)
	{
		err = clReleaseMemObject(dstMemB);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (commandQueue)
	{
		err = clReleaseCommandQueue(commandQueue);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseCommandQueue returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (device)
	{
		err = clReleaseDevice(device);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseDevice returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (context)
	{
		err = clReleaseContext(context);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseContext returned '%s'.\n", TranslateOpenCLError(err));
		}
	}

	/*
	* Note there is no procedure to deallocate platform
	* because it was not created at the startup,
	* but just queried from OpenCL runtime.
	*/
}

bool CheckPreferredPlatformMatch(cl_platform_id platform, const char* preferredPlatform)
{
	size_t stringLength = 0;
	cl_int err = CL_SUCCESS;
	bool match = false;

	// In order to read the platform's name, we first read the platform's name string length (param_value is NULL).
	// The value returned in stringLength
	err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &stringLength);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetPlatformInfo() to get CL_PLATFORM_NAME length returned '%s'.\n", TranslateOpenCLError(err));
		return false;
	}

	// Now, that we know the platform's name string length, we can allocate enough space before read it
	std::vector<char> platformName(stringLength);

	// Read the platform's name string
	// The read value returned in platformName
	err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, stringLength, &platformName[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetplatform_ids() to get CL_PLATFORM_NAME returned %s.\n", TranslateOpenCLError(err));
		return false;
	}

	// Now check if the platform's name is the required one
	if (strstr(&platformName[0], preferredPlatform) != 0)
	{
		// The checked platform is the one we're looking for
		match = true;
	}

	return match;
}


cl_platform_id FindOpenCLPlatform(const char* preferredPlatform, cl_device_type deviceType)
{
	cl_uint numPlatforms = 0;
	cl_int err = CL_SUCCESS;

	// Get (in numPlatforms) the number of OpenCL platforms available
	// No platform ID will be return, since platforms is NULL
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetplatform_ids() to get num platforms returned %s.\n", TranslateOpenCLError(err));
		return NULL;
	}
	LogInfo("Number of available platforms: %u\n", numPlatforms);

	if (0 == numPlatforms)
	{
		LogError("Error: No platforms found!\n");
		return NULL;
	}

	std::vector<cl_platform_id> platforms(numPlatforms);

	// Now, obtains a list of numPlatforms OpenCL platforms available
	// The list of platforms available will be returned in platforms
	err = clGetPlatformIDs(numPlatforms, &platforms[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetplatform_ids() to get platforms returned %s.\n", TranslateOpenCLError(err));
		return NULL;
	}

	// Check if one of the available platform matches the preferred requirements
	for (cl_uint i = 0; i < numPlatforms; i++)
	{
		bool match = true;
		cl_uint numDevices = 0;

		// If the preferredPlatform is not NULL then check if platforms[i] is the required one
		// Otherwise, continue the check with platforms[i]
		if ((NULL != preferredPlatform) && (strlen(preferredPlatform) > 0))
		{
			// In case we're looking for a specific platform
			match = CheckPreferredPlatformMatch(platforms[i], preferredPlatform);
		}

		// match is true if the platform's name is the required one or don't care (NULL)
		if (match)
		{
			// Obtains the number of deviceType devices available on platform
			// When the function failed we expect numDevices to be zero.
			// We ignore the function return value since a non-zero error code
			// could happen if this platform doesn't support the specified device type.
			err = clGetDeviceIDs(platforms[i], deviceType, 0, NULL, &numDevices);
			if (CL_SUCCESS != err)
			{
				LogError("clGetDeviceIDs() returned %s.\n", TranslateOpenCLError(err));
			}

			if (0 != numDevices)
			{
				// There is at list one device that answer the requirements
				return platforms[i];
			}
		}
	}

	return NULL;
}


int GetPlatformAndDeviceVersion(cl_platform_id platformId, ocl_args_d_t *ocl)
{
	cl_int err = CL_SUCCESS;

	// Read the platform's version string length (param_value is NULL).
	// The value returned in stringLength
	size_t stringLength = 0;
	err = clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, 0, NULL, &stringLength);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetPlatformInfo() to get CL_PLATFORM_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
		return err;
	}

	// Now, that we know the platform's version string length, we can allocate enough space before read it
	std::vector<char> platformVersion(stringLength);

	// Read the platform's version string
	// The read value returned in platformVersion
	err = clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, stringLength, &platformVersion[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetplatform_ids() to get CL_PLATFORM_VERSION returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	if (strstr(&platformVersion[0], "OpenCL 2.0") != NULL)
	{
		ocl->platformVersion = OPENCL_VERSION_2_0;
	}

	// Read the device's version string length (param_value is NULL).
	err = clGetDeviceInfo(ocl->device, CL_DEVICE_VERSION, 0, NULL, &stringLength);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
		return err;
	}

	// Now, that we know the device's version string length, we can allocate enough space before read it
	std::vector<char> deviceVersion(stringLength);

	// Read the device's version string
	// The read value returned in deviceVersion
	err = clGetDeviceInfo(ocl->device, CL_DEVICE_VERSION, stringLength, &deviceVersion[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_VERSION returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	if (strstr(&deviceVersion[0], "OpenCL 2.0") != NULL)
	{
		ocl->deviceVersion = OPENCL_VERSION_2_0;
	}

	// Read the device's OpenCL C version string length (param_value is NULL).
	err = clGetDeviceInfo(ocl->device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &stringLength);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_OPENCL_C_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
		return err;
	}

	// Now, that we know the device's OpenCL C version string length, we can allocate enough space before read it
	std::vector<char> compilerVersion(stringLength);

	// Read the device's OpenCL C version string
	// The read value returned in compilerVersion
	err = clGetDeviceInfo(ocl->device, CL_DEVICE_OPENCL_C_VERSION, stringLength, &compilerVersion[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_OPENCL_C_VERSION returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	else if (strstr(&compilerVersion[0], "OpenCL C 2.0") != NULL)
	{
		ocl->compilerVersion = OPENCL_VERSION_2_0;
	}


	// extra info
	clGetDeviceInfo(ocl->device, CL_DEVICE_NAME, 0, NULL, &stringLength);
	deviceName.resize(stringLength);
	clGetDeviceInfo(ocl->device, CL_DEVICE_NAME, stringLength, &deviceName[0], NULL);
	clGetDeviceInfo(ocl->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &deviceMem, NULL);
	clGetDeviceInfo(ocl->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &deviceMaxAlloc, NULL);


	return err;
}

int SetupOpenCL(ocl_args_d_t *ocl, cl_device_type deviceType)
{
	// The following variable stores return codes for all OpenCL calls.
	cl_int err = CL_SUCCESS;

	// Query for all available OpenCL platforms on the system
	// Here you enumerate all platforms and pick one which name has preferredPlatform as a sub-string
	cl_platform_id platformId = FindOpenCLPlatform(platName, deviceType);
	if (NULL == platformId)
	{
		LogError("Error: Failed to find OpenCL platform.\n");
		return CL_INVALID_VALUE;
	}

	// Create context with device of specified type.
	// Required device type is passed as function argument deviceType.
	// So you may use this function to create context for any CPU or GPU OpenCL device.
	// The creation is synchronized (pfn_notify is NULL) and NULL user_data
	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platformId, 0 };
	ocl->context = clCreateContextFromType(contextProperties, deviceType, NULL, NULL, &err);
	if ((CL_SUCCESS != err) || (NULL == ocl->context))
	{
		LogError("Couldn't create a context, clCreateContextFromType() returned '%s'.\n", TranslateOpenCLError(err));
		return err;
	}

	// Query for OpenCL device which was used for context creation
	size_t size;
	err = clGetContextInfo(ocl->context, CL_CONTEXT_DEVICES, sizeof(cl_device_id) * 8, &ocl->devices, &size);
	ocl->device = ocl->devices[devID];


	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetContextInfo() to get list of devices returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	// Read the OpenCL platform's version and the device OpenCL and OpenCL C versions
	GetPlatformAndDeviceVersion(platformId, ocl);

	// Create command queue.
	// OpenCL kernels are enqueued for execution to a particular device through special objects called command queues.
	// Command queue guarantees some ordering between calls and other OpenCL commands.
	// Here you create a simple in-order OpenCL command queue that doesn't allow execution of two kernels in parallel on a target device.
#ifdef CL_VERSION_2_0

	const cl_command_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,  0 };

	ocl->commandQueue = clCreateCommandQueueWithProperties(ocl->context, ocl->device, properties, &err);

#else
	// default behavior: OpenCL 1.2
	cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
	ocl->commandQueue = clCreateCommandQueue(ocl->context, ocl->device, properties, &err);
#endif
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateCommandQueue() returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	return CL_SUCCESS;
}

int ReadSourceFromFile(const char* fileName, char** source, size_t* sourceSize)
{
	int errorCode = CL_SUCCESS;

	FILE* fp = NULL;
	fopen_s(&fp, fileName, "rb");
	if (fp == NULL)
	{
		LogError("Error: Couldn't find program source file '%s'.\n", fileName);
		errorCode = CL_INVALID_VALUE;
	}
	else {
		fseek(fp, 0, SEEK_END);
		*sourceSize = ftell(fp);
		fseek(fp, 0, SEEK_SET);

		*source = new char[*sourceSize];
		if (*source == NULL)
		{
			LogError("Error: Couldn't allocate %d bytes for program source from file '%s'.\n", *sourceSize, fileName);
			errorCode = CL_OUT_OF_HOST_MEMORY;
		}
		else {
			fread(*source, 1, *sourceSize, fp);
		}
	}
	return errorCode;
}

int CreateAndBuildProgramCUCKOO(ocl_args_d_t *ocl)
{
	cl_int err = CL_SUCCESS;

	// Upload the OpenCL C source code from the input file to source
	// The size of the C program is returned in sourceSize
	char* source = NULL;
	size_t src_size = 0;
	err = ReadSourceFromFile("_lean.cl", &source, &src_size);
	if (CL_SUCCESS != err)
	{
		LogError("Error: ReadSourceFromFile returned %s.\n", TranslateOpenCLError(err));
		goto Finish;
	}

	// And now after you obtained a regular C string call clCreateProgramWithSource to create OpenCL program object.
	ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char**)&source, &src_size, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateProgramWithSource returned %s.\n", TranslateOpenCLError(err));
		goto Finish;
	}

	// Build the program
	// During creation a program is not built. You need to explicitly call build function.
	// Here you just use create-build sequence,
	// but there are also other possibilities when program consist of several parts,
	// some of which are libraries, and you may want to consider using clCompileProgram and clLinkProgram as
	// alternatives.
	err = clBuildProgram(ocl->program, 1, &ocl->device,
		strcmp("NVIDIA", platName) == 0 ?
		" -cl-std=CL2.0 -D NVIDIA \0" :
		(strcmp("AMD", platName) == 0 ? " -cl-std=CL2.0  -D AMD \0" : " -cl-std=CL2.0  -D INTEL \0"),
		NULL, NULL);

	if (CL_SUCCESS != err)
	{
		LogError("Error: clBuildProgram() for source program returned %s.\n", TranslateOpenCLError(err));

		// In case of error print the build log to the standard output
		// First check the size of the log
		// Then allocate the memory and obtain the log from the program
		if (err == CL_BUILD_PROGRAM_FAILURE)
		{
			size_t log_size = 0;
			clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

			std::vector<char> build_log(log_size);
			clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, log_size, &build_log[0], NULL);

			for (auto i : build_log)
				std::cerr << i << ' ';

			LogError("Error happened during the build of OpenCL program.\nBuild log:%s", &build_log[0]);
		}
	}

Finish:
	if (source)
	{
		delete[] source;
		source = NULL;
	}

	return err;
}