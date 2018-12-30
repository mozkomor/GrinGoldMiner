
namespace OpenCl.DotNetCore.Interop
{
    /// <summary>
    /// Represents an enumeration for the different types of information that can be queried from a command queue.
    /// </summary>
    public enum CommandQueueInformation : uint
    {
        /// <summary>
        /// The context specified when the command-queue is created.
        /// </summary>
        Context = 0x1090,

        /// <summary>
        /// The device specified when the command-queue is created.
        /// </summary>
        Device = 0x1091,

        /// <summary>
        /// The reference count should be considered immediately stale. It is unsuitable for general use in applications. This feature is provided for identifying memory leaks.
        /// </summary>
        ReferenceCount = 0x1092,

        /// <summary>
        /// The currently specified properties for the command-queue.
        /// </summary>
        Properties = 0x1093,

        /// <summary>
        /// The currently specified size for the device command-queue. This query is only supported for device command queues.
        /// </summary>
        Size = 0x1094,

        /// <summary>
        /// The current default command queue for the underlying device.
        /// </summary>
        DeviceDefault = 0x1095
    }
}