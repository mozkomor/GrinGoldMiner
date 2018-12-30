
#region Using Directives

using System;

#endregion

namespace OpenCl.DotNetCore.Interop.Memory
{
    /// <summary>
    /// Represents the memory flags, that are used to create memory buffers.
    /// </summary>
    [Flags]
    public enum MemoryFlag : ulong
    {
        /// <summary>
        /// This flag specifies that the memory object will be read and written by a kernel. This is the default.
        /// </summary>
        ReadWrite = 1 << 0,

        /// <summary>
        /// This flags specifies that the memory object will be written but not read by a kernel. Reading from a buffer or image object created with <see cref="WriteOnly"/> inside a kernel is undefined. <see cref="ReadWrite"/> and
        /// <see cref="WriteOnly"/> are mutually exclusive.
        /// </summary>
        WriteOnly = 1 << 1,

        /// <summary>
        /// This flag specifies that the memory object is a read-only memory object when used inside a kernel. Writing to a buffer or image object created with <see cref="ReadOnly"/> inside a kernel is undefined. <see cref="ReadWrite"/> or
        /// <see cref="WriteOnly"/> and <see cref="ReadOnly"/> are mutually exclusive.
        /// </summary>
        ReadOnly = 1 << 2,

        /// <summary>
        /// This flag is valid only if a host pointer was specified. If specified, it indicates that the application wants the OpenCL implementation to use memory referenced by the specified host pointer as the storage bits for the memory
        /// object. OpenCL implementations are allowed to cache the buffer contents pointed to by the specified host pointer in device memory. This cached copy can be used when kernels are executed on a device. The result of OpenCL commands
        /// that operate on multiple buffer objects created with the same host pointer or overlapping host regions is considered to be undefined. Refer to the description of the alignment rules for the specified host pointer for memory objects
        /// (buffer and images) created using <see cref="UseHostPointer"/>.
        /// </summary>
        UseHostPointer = 1 << 3,

        /// <summary>
        /// This flag specifies that the application wants the OpenCL implementation to allocate memory from host accessible memory. <see cref="AllocateHostPointer"/> and <see cref="UseHostPointer"/> are mutually exclusive.
        /// </summary>
        AllocateHostPointer = 1 << 4,

        /// <summary>
        /// This flag is valid only if a host pointer was specified. If specified, it indicates that the application wants the OpenCL implementation to allocate memory for the memory object and copy the data from memory referenced by the
        /// specified host pointer. <see cref="CopyHostPointer"/> and <see cref="UseHostPointer"/> are mutually exclusive. <see cref="CopyHostPointer"/> can be used with <see cref="AllocateHostPointer"/> to initialize the contents of the
        /// memory object allocated using host-accessible (e.g. PCIe) memory.
        /// </summary>
        CopyHostPointer = 1 << 5,

        /// <summary>
        /// This flag specifies that the host will only write to the memory object (using OpenCL APIs that enqueue a write or a map for write). This can be used to optimize write access from the host (e.g. enable write combined allocations for
        /// memory objects for devices that communicate with the host over a system bus such as PCIe).
        /// </summary>
        HostWriteOnly = 1 << 7,

        /// <summary>
        /// This flag specifies that the host will only read the memory object (using OpenCL APIs that enqueue a read or a map for read). <see cref="HostWriteOnly"/> and <see cref="HostReadOnly"/> are mutually exclusive.
        /// </summary>
        HostReadOnly = 1 << 8,

        /// <summary>
        /// This flag specifies that the host will not read or write the memory object. <see cref="HostWriteOnly"/> or <see cref="HostReadOnly"/> and <see cref="HostNoAccess"/> are mutually exclusive.
        /// </summary>
        HostNoAccess = 1 << 9,

        /// <summary>
        /// Can be used to get a list of supported image formats that can be both read from and written to by a kernel.
        /// </summary>
        KernelReadAndWrite = 1 << 12
    }
}