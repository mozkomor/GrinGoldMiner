
#region Using Directives

using System;

#endregion

namespace OpenCl.DotNetCore.Interop.EnqueuedCommands
{
    /// <summary>
    /// Represents an enumeration for the different flags, that can be used when mapping device memory to host memory.
    /// </summary>
    [Flags]
    public enum MapFlag : ulong
    {
        /// <summary>
        /// This flag specifies that the region being mapped in the memory object is being mapped for reading. The pointer returned by <see cref="EnqueueMapBuffer"/> and <see cref="EnqueueMapImage"/> is guaranteed to contain the latest bits in the
        /// region being mapped when the <see cref="EnqueueMapBuffer"/> or <see cref="EnqueueMapImage"/> command has completed.
        /// </summary>
        Read = 1 << 0,

        /// <summary>
        /// This flag specifies that the region being mapped in the memory object is being mapped for writing. The pointer returned by <see cref="EnqueueMapBuffer"/> and <see cref="EnqueueMapImage"/> is guaranteed to contain the latest bits in the
        /// region being mapped when the <see cref="EnqueueMapBuffer"/> or <see cref="EnqueueMapImage"/> command has completed.
        /// </summary>
        Write = 1 << 1,

        /// <summary>
        /// This flag specifies that the region being mapped in the memory object is being mapped for writing. The contents of the region being mapped are to be discarded. This is typically the case when the region being mapped is overwritten by
        /// the host. This flag allows the implementation to no longer guarantee that the pointer returned by <see cref="EnqueueMapBuffer"/> or <see cref="EnqueueMapImage"/> contains the latest bits in the region being mapped which can be a
        /// significant performance enhancement. <c>MapFlag.Read</c> or <c>MapFlag.Write</c> and <c>MapFlag.WriteInvalidateRegion</c> are mutually exclusive.
        /// </summary>
        WriteInvalidateRegion = 1 << 2
    }
}