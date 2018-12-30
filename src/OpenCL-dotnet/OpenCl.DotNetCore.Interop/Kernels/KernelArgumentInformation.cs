
namespace OpenCl.DotNetCore.Interop.Kernels
{
    /// <summary>
    /// Represents an enumeration for the different types of information that can be queried from an OpenCL kernel parameter.
    /// </summary>
    public enum KernelArgumentInformation : uint
    {
        /// <summary>
        /// The address qualifier specified for the argument.
        /// </summary>
        AddressQualifier = 0x1196,

        /// <summary>
        /// The access qualifier specified for the argument.
        /// </summary>
        AccessQualifier = 0x1197,

        /// <summary>
        /// The type name specified for the argument. The type name returned will be the argument type name as it was declared with any whitespace removed. If argument type name is an unsigned scalar type (i.e. unsigned char, unsigned short,
        /// unsigned int, unsigned long), uchar, ushort, uint and ulong will be returned. The argument type name returned does not include any type qualifiers.
        /// </summary>
        TypeName = 0x1198,

        /// <summary>
        /// The type qualifier specified for the argument.
        /// </summary>
        TypeQualifier = 0x1199,

        /// <summary>
        /// The name specified for the argument.
        /// </summary>
        Name = 0x119A
    }
}