
#region Using Directives

using System;
using System.Runtime.InteropServices;

#endregion

namespace OpenCl.DotNetCore.Interop.SvmAllocations
{
    /// <summary>
    /// Represents a wrapper for the native methods of the OpenCL SVM Allocations API.
    /// </summary>
    public static class SvmAllocationsNativeApi
    {
        #region Public Static Methods

        /// <summary>
        /// Allocates a shared virtual memory (SVM) buffer that can be shared by the host and all devices in an OpenCL context that support shared virtual memory.
        /// </summary>
        /// <param name="context">A valid OpenCL context used to create the SVM buffer.</param>
        /// <param name="flags">An enumeration that is used to specify allocation and usage information.</param>
        /// <param name="size">The size in bytes of the SVM buffer to be allocated.</param>
        /// <param name="alignment">
        /// The minimum alignment in bytes that is required for the newly created buffer’s memory region. It must be a power of two up to the largest data type supported by the OpenCL device. For the full profile, the largest data type is
        /// long16. For the embedded profile, it is long16 if the device supports 64-bit integers; otherwise it is int16. If alignment is 0, a default alignment will be used that is equal to the size of largest data type supported by the
        /// OpenCL implementation.
        /// </param>
        /// <returns>
        /// Returns a valid non-<c>null</c> shared virtual memory address if the SVM buffer is successfully allocated. Otherwise, like <c>malloc</c>, it returns a <c>null</c> pointer value. <see cref="SvmAllocate"/> will fail if:
        /// 
        /// - <see cref="context"/> is not a valid context
        /// - <see cref="flags"/> does not contain <c>SvmMemoryFlag.SvmFineGrainBuffer</c> but does contain <c>SvmMemoryFlag.SvmAtomics</c>
        /// - Values specified in <see cref="flags"/> do not follow rules described for supported values
        /// - <c>SvmMemoryFlag.SvmFineGrainBuffer</c> or <c>SvmMemoryFlag.SvmAtomics</c> is specified in <see cref="flags"/> and these are not supported by at least one device in <see cref="context"/>
        /// - The values specified in <see cref="flags"/> are not valid
        /// - <see cref="size"/> is 0 or greater than the <c>DeviceInformation.MaximumMemoryAllocationSize</c> value for any device in <see cref="context"/>
        /// - <see cref="alignment"/> is not a power of two or the OpenCL implementation cannot support the specified alignment for at least one device in <see cref="context"/>
        /// - There was a failure to allocate resources
        /// </returns>
        [IntroducedInOpenCl(2, 0)]
        [DllImport("OpenCL", EntryPoint = "clSVMAlloc")]
        public static extern IntPtr SvmAllocate(
            [In] IntPtr context,
            [In] [MarshalAs(UnmanagedType.U8)] SvmMemoryFlag flags,
            [In] UIntPtr size,
            [In] [MarshalAs(UnmanagedType.U4)] uint alignment
        );

        /// <summary>
        /// Frees a shared virtual memory buffer allocated using <see cref="SvmAllocate"/>. Note that <see cref="SvmFree"/> does not wait for previously enqueued commands that may be using <see cref="svmPointer"/> to finish before freeing
        /// <see cref="svmPointer"/>. It is the responsibility of the application to make sure that enqueued commands that use <see cref="svmPointer"/> have finished before freeing <see cref="svmPointer"/>.
        /// </summary>
        /// <param name="context">A valid OpenCL context used to create the SVM buffer.</param>
        /// <param name="svmPointer">Must be the value returned by a call to <see cref="SvmAllocate"/>. If a <c>null</c> pointer is passed in <see cref="svmPointer"/>, no action occurs.</param>
        [IntroducedInOpenCl(2, 0)]
        [DllImport("OpenCL", EntryPoint = "clSVMFree")]
        public static extern void SvmFree(
            [In] IntPtr context,
            [In] IntPtr svmPointer
        );

        #endregion
    }
}