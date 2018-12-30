
namespace OpenCl.DotNetCore.Platforms
{
    /// <summary>
    /// Represents an enumeration for the different profiles that can be supported by OpenCL platforms.
    /// </summary>
    public enum Profile
    {
        /// <summary>
        /// The full OpenCL specification is supported.
        /// </summary>
        Full,

        /// <summary>
        /// A subset of the OpenCL specification for embedded devices is supported.
        /// </summary>
        Embedded
    }
}