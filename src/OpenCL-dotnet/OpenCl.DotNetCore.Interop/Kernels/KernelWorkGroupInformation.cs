
namespace OpenCl.DotNetCore.Interop.Kernels
{
    /// <summary>
    /// Represents an enumeration for the different types of information that can be queried from an OpenCL kernel work group.
    /// </summary>
    public enum KernelWorkGroupInformation : uint
    {
        /// <summary>
        /// This provides a mechanism for the application to query the maximum work-group size that can be used to execute the kernel on a specific device. The OpenCL implementation uses the resource requirements of the kernel (register usage
        /// etc.) to determine what this work-group size should be. As a result and unlike <c>DeviceInformation.MaximumWorkGoupSize</c> this value may vary from one kernel to another as well as one device to another.
        /// <c>KernelWorkGroupInformation.WorkGroupSize</c> will be less than or equal to <c>DeviceInformation.MaximumWorkGoupSize</c> for a given kernel object.
        /// </summary>
        WorkGroupSize = 0x11B0,

        /// <summary>
        /// Returns the work-group size specified in the kernel source or IL. If the work-group size is not specified in the kernel source or IL, (0, 0, 0) is returned.
        /// </summary>
        CompileWorkGroupSize = 0x11B1,

        /// <summary>
        /// Returns the amount of local memory in bytes being used by a kernel. This includes local memory that may be needed by an implementation to execute the kernel, variables declared inside the kernel with the <c>__local</c> address
        /// qualifier and local memory to be allocated for arguments to the kernel declared as pointers with the <c>__local</c> address qualifier and whose size is specified with <see cref="SetKernelArgument"/>. If the local memory size, for
        /// any pointer argument to the kernel declared with the <c>__local</c> address qualifier, is not specified, its size is assumed to be 0.
        /// </summary>
        LocalMemorySize = 0x11B2,

        /// <summary>
        /// Returns the preferred multiple of workgroup size for launch. This is a performance hint. Specifying a workgroup size that is not a multiple of the value returned by this query as the value of the local work size argument to
        /// <see cref="EnqueueNDRangeKernel"/> will not fail to enqueue the kernel for execution unless the work-group size specified is larger than the device maximum.
        /// </summary>
        PreferredWorkGroupSizeMultiple = 0x11B3,

        /// <summary>
        /// Returns the minimum amount of private memory, in bytes, used by each work-item in the kernel. This value may include any private memory needed by an implementation to execute the kernel, including that used by the language built-ins
        /// and variable declared inside the kernel with the <c>__private</c> qualifier.
        /// </summary>
        PrivateMemorySize = 0x11B4,

        /// <summary>
        /// This provides a mechanism for the application to query the maximum global size that can be used to execute a kernel (i.e. <see cref="globalWorkSize"/> argument to <see cref="EnqueueNDRangeKernel"/>) on a custom device given by
        /// device or a built-in kernel on an OpenCL device. If the device is not a custom device or the kernel is not a built-in kernel, <see cref="GetKernelWorkGroupInformation"/> returns the error <c>Result.InvalidValue</c>.
        /// </summary>
        GlobalWorkSize = 0x11B5
    }
}