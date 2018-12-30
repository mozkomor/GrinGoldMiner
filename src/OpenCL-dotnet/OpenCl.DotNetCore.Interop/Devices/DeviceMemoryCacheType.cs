
namespace OpenCl.DotNetCore.Interop.Devices
{
    /// <summary>
    /// Represents an enumeration for the different types of device memory caches.
    /// </summary>
    public enum DeviceMemoryCacheType : uint
    {
        /// <summary>
        /// No device memory cache is available.
        /// </summary>
        None = 0x0,

        /// <summary>
        /// The device memory cache is read-only.
        /// </summary>
        ReadOnlyCache = 0x1,

        /// <summary>
        /// The device memory cache is readable and writable.
        /// </summary>
        ReadWriteCache = 0x2
    }
}