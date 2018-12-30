
namespace OpenCl.DotNetCore.Interop
{
    /// <summary>
    /// Represents an enumeration for the result codes that are returned by the native OpenCL functions.
    /// </summary>
    public enum Result : int
    {
        #region OpenCL Error Codes

        /// <summary>
        /// The function is executed successfully.
        /// </summary>
        Success = 0,

        /// <summary>
        /// No device that matched the specified device type could be found.
        /// </summary>
        DeviceNotFound = -1,

        /// <summary>
        /// The device is currently not available.
        /// </summary>
        DeviceNotAvailable = -2,

        /// <summary>
        /// The compiler is not available for the current platform.
        /// </summary>
        CompilerNotAvailable = -3,

        /// <summary>
        /// 
        /// </summary>
        MemObjectAllocationFailure = -4,

        /// <summary>
        /// There was a failure to allocate resources required by the OpenCL implementation on the on the device.
        /// </summary>
        OutOfResources = -5,

        /// <summary>
        /// There was a failure to allocate resources required by the OpenCL implementation on the host.
        /// </summary>
        OutOfHostMemory = -6,

        /// <summary>
        /// 
        /// </summary>
        ProfilingInformationNotAvailable = -7,

        /// <summary>
        /// 
        /// </summary>
        MemCopyOverlap = -8,

        /// <summary>
        /// 
        /// </summary>
        ImageFormatMismatch = -9,

        /// <summary>
        /// 
        /// </summary>
        ImageFormatNotSupported = -10,

        /// <summary>
        /// If there is a failure to build the program executable.
        /// </summary>
        BuildProgramFailure = -11,

        /// <summary>
        /// 
        /// </summary>
        MapFailure = -12,

        /// <summary>
        /// One or more of the provided arguments has an invalid value.
        /// </summary>
        InvalidValue = -30,

        /// <summary>
        /// The specified device type is not a valid device type.
        /// </summary>
        InvalidDeviceType = -31,

        /// <summary>
        /// The specified platform is not a valid platform.
        /// </summary>
        InvalidPlatform = -32,

        /// <summary>
        /// 
        /// </summary>
        InvalidDevice = -33,

        /// <summary>
        /// 
        /// </summary>
        InvalidContext = -34,

        /// <summary>
        /// The specified queue properties are not valid.
        /// </summary>
        InvalidQueueProperties = -35,

        /// <summary>
        /// 
        /// </summary>
        InvalidCommandQueue = -36,

        /// <summary>
        /// 
        /// </summary>
        InvalidHostPtr = -37,

        /// <summary>
        /// 
        /// </summary>
        InvalidMemObject = -38,

        /// <summary>
        /// 
        /// </summary>
        InvalidImageFormatDescriptor = -39,

        /// <summary>
        /// 
        /// </summary>
        InvalidImageSize = -40,

        /// <summary>
        /// 
        /// </summary>
        InvalidSampler = -41,

        /// <summary>
        /// 
        /// </summary>
        InvalidBinary = -42,

        /// <summary>
        /// 
        /// </summary>
        InvalidBuildOptions = -43,

        /// <summary>
        /// The program is not a valid program object.
        /// </summary>
        InvalidProgram = -44,

        /// <summary>
        /// There is no successfully built executable program.
        /// </summary>
        InvalidProgramExecutable = -45,

        /// <summary>
        /// The specified kernel name was not found in the program.
        /// </summary>
        InvalidKernelName = -46,

        /// <summary>
        /// The function definition for the __kernel function such as the number of arguments or the argument types are not the same for all devices for which the program executable has been built.
        /// </summary>
        InvalidKernelDefinition = -47,

        /// <summary>
        /// 
        /// </summary>
        InvalidKernel = -48,

        /// <summary>
        /// 
        /// </summary>
        InvalidArgIndex = -49,

        /// <summary>
        /// 
        /// </summary>
        InvalidArgValue = -50,

        /// <summary>
        /// 
        /// </summary>
        InvalidArgSize = -51,

        /// <summary>
        /// 
        /// </summary>
        InvalidKernelArgs = -52,

        /// <summary>
        /// 
        /// </summary>
        InvalidWorkDimension = -53,

        /// <summary>
        /// 
        /// </summary>
        InvalidWorkGroupSize = -54,

        /// <summary>
        /// 
        /// </summary>
        InvalidWorkItemSize = -55,

        /// <summary>
        /// 
        /// </summary>
        InvalidGlobalOffset = -56,

        /// <summary>
        /// 
        /// </summary>
        InvalidEventWaitList = -57,

        /// <summary>
        /// 
        /// </summary>
        InvalidEvent = -58,

        /// <summary>
        /// The operation performed is invalid.
        /// </summary>
        InvalidOperation = -59,

        /// <summary>
        /// 
        /// </summary>
        InvalidGLObject = -60,

        /// <summary>
        /// 
        /// </summary>
        InvalidBufferSize = -61,

        /// <summary>
        /// 
        /// </summary>
        InvalidMipLevel = -62,

        /// <summary>
        /// 
        /// </summary>
        InvalidGlobalWorkSize = -63,

        /// <summary>
        /// 
        /// </summary>
        InvalidProperty = -64,

        /// <summary>
        /// 
        /// </summary>
        InvalidImageDescriptor = -65,

        /// <summary>
        /// 
        /// </summary>
        InvalidCompilerOptions = -66,
        
        /// <summary>
        /// 
        /// </summary>
        InvalidLinkerOptions = -67,
        
        /// <summary>
        /// 
        /// </summary>
        InvalidDevicePartitionCount = -68,
        
        /// <summary>
        /// 
        /// </summary>
        InvalidPipeSize = -69,

        /// <summary>
        /// 
        /// </summary>
        InvalidDeviceQueue = -70,

        #endregion

        #region OpenCL Extension Additional Error Codes

        /// <summary>
        /// If the cl_khr_icd extension is enabled and no platforms are found.
        /// </summary>
        PlatformNotFoundKhr = -1001,

        /// <summary>
        /// 
        /// </summary>
        DevicePartitionFailedExt = -1057,

        /// <summary>
        /// 
        /// </summary>
        InvalidPartitionCoundExt = -1058,

        /// <summary>
        /// 
        /// </summary>
        InvalidPartitionNameExt = -1059

        #endregion
    }
}