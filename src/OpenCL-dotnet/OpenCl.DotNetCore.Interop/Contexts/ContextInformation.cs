
namespace OpenCl.DotNetCore.Interop
{
    /// <summary>
    /// Represents an enumeration for the different types of information that can be queried from a context.
    /// </summary>
    public enum ContextInformation : uint
    {
        /// <summary>
        /// The context reference count. The reference count returned should be considered immediately stale. It is unsuitable for general use in applications. This feature is provided for identifying memory leaks.
        /// </summary>
        ReferenceCount = 0x1080,

        /// <summary>
        /// The list of devices in the context.
        /// </summary>
        Devices = 0x1081,

        /// <summary>
        /// The properties argument specified in <see cref="CreateContext"/> or <see cref="CreateContextFromType"/>.
        /// </summary>
        Properties = 0x1082,

        /// <summary>
        /// The number of devices in context.
        /// </summary>
        NumberOfDevices = 0x1083
    }
}