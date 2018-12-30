
namespace OpenCl.DotNetCore.Interop.Programs
{
    /// <summary>
    /// Represents an enumeration for the different types of information that can be queried from a program build.
    /// </summary>
    public enum ProgramBuildInformation : uint
    {
        /// <summary>
        /// The build, compile, or link status.
        /// </summary>
        Status = 0x1181,

        /// <summary>
        /// The build, compile, or link options.
        /// </summary>
        Options = 0x1182,

        /// <summary>
        /// The build, compile, or link log.
        /// </summary>
        Log = 0x1183,

        /// <summary>
        /// The program binary type for the device.
        /// </summary>
        BinaryType = 0x1184,

        /// <summary>
        /// The total amount of storage, in bytes, used by program variables in the global address space.
        /// </summary>
        GlobalVariableTotalSize = 0x1185
    }
}