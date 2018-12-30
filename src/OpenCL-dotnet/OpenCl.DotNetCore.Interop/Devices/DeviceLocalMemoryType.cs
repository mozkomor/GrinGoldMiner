
namespace OpenCl.DotNetCore.Interop.Devices
{
    /// <summary>
    /// Represents an enumeration for the different type of local device memory.
    /// </summary>
    public enum DeviceLocalMemoryType : uint
    {
        /// <summary>
        /// Custom devices may have no local memory support.
        /// </summary>
        None = 0x0,

        /// <summary>
        /// Dedicated local memory storage such as SRAM.
        /// </summary>
        Local = 0x1,

        /// <summary>
        /// No dedicated local memory storage is available, but rather global memory is used.
        /// </summary>
        Global = 0x2
    }
}