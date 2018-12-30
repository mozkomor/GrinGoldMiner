
namespace OpenCl.DotNetCore.Interop.Samplers
{
    /// <summary>
    /// Represents an enumeration for the different addressing modes that can be used for samplers.
    /// </summary>
    public enum AddressingMode : uint
    {
        /// <summary>
        /// No addressing mode.
        /// </summary>
        None = 0x1130,

        /// <summary>
        /// Clamps the image to the edge.
        /// </summary>
        ClampToEdge = 0x1131,

        /// <summary>
        /// Clamps the image.
        /// </summary>
        Clamp = 0x1132,

        /// <summary>
        /// Repeats the image.
        /// </summary>
        Repeat = 0x1133,

        /// <summary>
        /// Repeats the image mirrored.
        /// </summary>
        MirroredRepeat = 0x1134
    }
}