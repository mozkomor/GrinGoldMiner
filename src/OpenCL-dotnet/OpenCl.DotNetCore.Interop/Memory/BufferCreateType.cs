
namespace OpenCl.DotNetCore.Interop.Memory
{
    /// <summary>
    /// Represents an enumeration for the buffer creation type.
    /// </summary>
    public enum BufferCreateType : uint
    {

        /// <summary>
        /// Creates a buffer object that represents a specific region in the buffer.
        /// </summary>
        Region = 0x1220
    }
}