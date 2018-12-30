
namespace OpenCl.DotNetCore.Interop.Kernels
{
    /// <summary>
    /// Represents an enumeration for the different types of information that can be set for the kernel execution.
    /// </summary>
    public enum KernelExecutionInformation : uint
    {
        /// <summary>
        /// SVM pointers used by a kernel which are not passed as arguments to kernel. These addresses may be defined in SVM buffer(s) that are passed as arguments to kernel. These non-argument SVM pointers must be specified using
        /// <see cref="SetKernelExecutionInformation"/> for coarse-grain and fine-grain buffer SVM allocations but not for fine-grain system SVM allocations.
        /// </summary>
        SvmPointers = 0x11B6,

        /// <summary>
        /// This flag indicates whether the kernel uses pointers that are fine grain system SVM allocations. These fine grain system SVM pointers may be passed as arguments or defined in SVM buffers that are passed as arguments to the kernel.
        /// </summary>
        SvmFineGrainSystem = 0x11B7
    }
}