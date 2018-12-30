
namespace OpenCl.DotNetCore.Interop.Samplers
{
    /// <summary>
    /// Represents an enumeration for the different filter modes that can be used for samplers.
    /// </summary>
    public enum FilterMode : uint
    {
        /// <summary>
        /// The filter mode is nearest.
        /// </summary>
        Nearest = 0x1140,

        /// <summary>
        /// The filter mode is linear.
        /// </summary>
        Linear = 0x1141
    }
}