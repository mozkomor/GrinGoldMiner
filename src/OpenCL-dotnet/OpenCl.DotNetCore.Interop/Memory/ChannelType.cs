
namespace OpenCl.DotNetCore.Interop.Memory
{
    /// <summary>
    /// Represents an enumeration for the different data types that can be used for color channels in an image.
    /// </summary>
    public enum ChannelType : uint
    {
        /// <summary>
        /// Each channel component is a normalized signed 8-bit integer value.
        /// </summary>
        NormalizedSignedInteger8 = 0x10D0,

        /// <summary>
        /// Each channel component is a normalized signed 16-bit integer value.
        /// </summary>
        NormalizedSignedInteger16 = 0x10D1,

        /// <summary>
        /// Each channel component is a normalized unsigned 8-bit integer value.
        /// </summary>
        NormalizedUnsignedInteger8 = 0x10D2,

        /// <summary>
        /// Each channel component is a normalized unsigned 16-bit integer value.
        /// </summary>
        NormalizedUnsignedInteger16 = 0x10D3,

        /// <summary>
        /// Represents a normalized 5-6-5 3-channel RGB image. The channel order must be <c>ChannelOrder.Rgb</c> or <c>ChannelOrder.Rgbx</c>.
        /// </summary>
        NormalizedUnsignedShortFloat565 = 0x10D4,

        /// <summary>
        /// Represents a normalized x-5-5-5 4-channel xRGB image. The channel order must be <c>ChannelOrder.Rgb</c> or <c>ChannelOrder.Rgbx</c>.
        /// </summary>
        NormalizedUnsignedShortFloat555 = 0x10D5,

        /// <summary>
        /// Represents a normalized x-10-10-10 4-channel xRGB image. The channel order must be <c>ChannelOrder.Rgb</c> or <c>ChannelOrder.Rgbx</c>.
        /// </summary>
        NormalizedUnsignedInteger101010 = 0x10D6,

        /// <summary>
        /// Each channel component is an unnormalized signed 8-bit integer value.
        /// </summary>
        SignedInteger8 = 0x10D7,

        /// <summary>
        /// Each channel component is an unnormalized signed 16-bit integer value.
        /// </summary>
        SignedInteger16 = 0x10D8,

        /// <summary>
        /// Each channel component is an unnormalized signed 32-bit integer value.
        /// </summary>
        SignedInteger32 = 0x10D9,

        /// <summary>
        /// Each channel component is an unnormalized unsigned 8-bit integer value.
        /// </summary>
        UnsignedInteger8 = 0x10DA,

        /// <summary>
        /// Each channel component is an unnormalized unsigned 16-bit integer value.
        /// </summary>
        UnsignedInteger16 = 0x10DB,

        /// <summary>
        /// Each channel component is an unnormalized unsigned 32-bit integer value.
        /// </summary>
        UnsignedInteger32 = 0x10DC,

        /// <summary>
        /// Each channel component is a 16-bit half-float value.
        /// </summary>
        HalfFloat = 0x10DD,

        /// <summary>
        /// Each channel component is a single precision floating-point value.
        /// </summary>
        Float = 0x10DE,

        /// <summary>
        /// Each channel component is a normalized unsigned 24-bit integer value.
        /// </summary>
        NormalizedUnsignedInteger24 = 0x10DF,

        /// <summary>
        /// Represents a different version of a normalized x-10-10-10 4-channel xRGB image. The channel order must be <c>ChannelOrder.Rgb</c> or <c>ChannelOrder.Rgbx</c>.
        /// </summary>
        NormalizedUnsignedInteger101010Version2 = 0x10E0
    }
}