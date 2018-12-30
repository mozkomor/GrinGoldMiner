
#region Using Directives

using System;

#endregion

namespace OpenCl.DotNetCore.Interop.Devices
{
    /// <summary>
    /// Represents an enumration for the different SVM capabilities a device may have.
    /// </summary>
    [Flags]
    public enum DeviceSvmCapabilities : ulong
    {
        /// <summary>
        /// Support for coarse-grain buffer sharing using <see cref="SvmAllocate"/>. Memory consistency is guaranteed at synchronization points and the host must use calls to <see cref="EnqueueMapBuffer"/> and
        /// <see cref="EnqueueUnmapMemoryObject"/>.
        /// </summary>
        CoarseGrainBuffer = 1 << 0,

        /// <summary>
        /// Support for fine-grain buffer sharing using <see cref="SvmAllocate"/>. Memory consistency is guaranteed at synchronization points without need for <see cref="EnqueueMapBuffer"/> and <see cref="EnqueueUnmapMemoryObject"/>.
        /// </summary>
        FineGrainBuffer = 1 << 1,

        /// <summary>
        /// Support for sharing the host’s entire virtual memory including memory allocated using malloc. Memory consistency is guaranteed at synchronization points.
        /// </summary>
        FineGrainSystem = 1 << 2,

        /// <summary>
        /// Support for the OpenCL 2.0 atomic operations that provide memory consistency across the host and all OpenCL devices supporting fine-grain SVM allocations.
        /// </summary>
        Atomics = 1 << 3
    }
}