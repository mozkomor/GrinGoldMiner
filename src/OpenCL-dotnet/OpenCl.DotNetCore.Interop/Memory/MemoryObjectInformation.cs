
namespace OpenCl.DotNetCore.Interop.Memory
{
    /// <summary>
    /// Represents an enumeration for the different types of information that can be queried from an memory object.
    /// </summary>
    public enum MemoryObjectInformation : uint
    {
        /// <summary>
        /// The type of the memory object.
        /// </summary>
        Type = 0x1100,

        /// <summary>
        /// The flags argument value specified when the memory object was created.
        /// </summary>
        Flags = 0x1101,

        /// <summary>
        /// The actual size of the data store associated with memory object in bytes.
        /// </summary>
        Size = 0x1102,

        /// <summary>
        /// The host pointer argument value specified when memory object is created.
        /// </summary>
        HostPointer = 0x1103,

        /// <summary>
        /// The map count. The map count returned should be considered immediately stale. It is unsuitable for general use in applications. This feature is provided for debugging.
        /// </summary>
        MapCount = 0x1104,

        /// <summary>
        /// The reference count. The reference count returned should be considered immediately stale. It is unsuitable for general use in applications. This feature is provided for identifying memory leaks.
        /// </summary>
        ReferenceCount = 0x1105,

        /// <summary>
        /// The context specified when the memory object was created.
        /// </summary>
        Context = 0x1106,

        /// <summary>
        /// The memory object from which the memory object was created.
        /// </summary>
        AssociatedMemoryObject = 0x1107,

        /// <summary>
        /// The offset if memory object is a sub-buffer object created using <see cref="CreateSubBuffer"/>.
        /// </summary>
        Offset = 0x1108,

        /// <summary>
        /// <c>true</c> if the memory object is a buffer object that was created with <c>MemoryFlag.UseHostPointer</c> or is a subbuffer object of a buffer object that was created with <c>MemoryFlag.UseHostPointer</c> and the
        /// host pointer specified when the buffer object was created is a SVM pointer. Otherwise <c>false</c>.
        /// </summary>
        UsesSvmPointer = 0x1109
    }
}