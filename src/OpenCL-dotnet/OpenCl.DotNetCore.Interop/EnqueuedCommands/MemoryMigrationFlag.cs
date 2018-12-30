
#region Using Directives

using System;

#endregion

namespace OpenCl.DotNetCore.Interop.EnqueuedCommands
{
    /// <summary>
    /// Represents an enumeration for the different flags, that can be used when migrating memory.
    /// </summary>
    [Flags]
    public enum MemoryMigrationFlag : ulong
    {
        /// <summary>
        /// This flag indicates that the specified set of memory objects are to be migrated to the host, regardless of the target command-queue.
        /// </summary>
        Host = 1 << 0,

        /// <summary>
        /// This flag indicates that the contents of the set of memory objects are undefined after migration. The specified set of memory objects are migrated to the device associated with the command queue without incurring the overhead of
        /// migrating their contents.
        /// </summary>
        ContentUndefined = 1 << 1
    }
}