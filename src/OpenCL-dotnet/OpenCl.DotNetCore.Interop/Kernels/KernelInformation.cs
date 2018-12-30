
namespace OpenCl.DotNetCore.Interop.Kernels
{
    /// <summary>
    /// Represents an enumeration for the different types of information that can be queried from an OpenCL kernel.
    /// </summary>
    public enum KernelInformation : uint
    {
        /// <summary>
        /// The kernel function name.
        /// </summary>
        FunctionName = 0x1190,
        
        /// <summary>
        /// The number of arguments to kernel.
        /// </summary>
        NumberOfArguments = 0x1191,

        /// <summary>
        /// The kernel reference count. The reference count returned should be considered immediately stale. It is unsuitable for general use in applications. This feature is provided for identifying memory leaks.
        /// </summary>
        ReferenceCount = 0x1192,

        /// <summary>
        /// The context associated with kernel.
        /// </summary>
        Context = 0x1193,

        /// <summary>
        /// The program object associated with kernel.
        /// </summary>
        Program = 0x1194,

        /// <summary>
        /// Attributes specified using the __attribute__ qualifier with the kernel function declaration in the program source. These attributes include those on the __attribute__ page and other attributes supported by an implementation.
        /// Attributes are returned as they were declared inside __attribute__((...)), with any surrounding whitespace and embedded newlines removed. When multiple attributes are present, they are returned as a single, space delimited string.
        /// For kernels not created from OpenCL C source and the <see cref="CreateProgramWithSource"/> API call the string returned from this query will be empty.
        /// </summary>
        Attributes = 0x1195,

        /// <summary>
        /// This provides a mechanism for the application to query the maximum number of sub-groups that may make up each work-group to execute a kernel on a specific device. The OpenCL implementation uses the resource requirements of the
        /// kernel (register usage etc.) to determine what this work-group size should be. The returned value may be used to compute a work-group size to enqueue the kernel with to give a round number of sub-groups for an enqueue.
        /// </summary>
        MaxNumberOfSubGroups = 0x11B9,

        /// <summary>
        /// Returns the number of sub-groups specified in the kernel source or IL. If the sub-group count is not specified then 0 is returned.
        /// </summary>
        CompileNumberOfSubGroups = 0x11BA
    }
}