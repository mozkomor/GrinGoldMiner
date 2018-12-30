
namespace OpenCl.DotNetCore.Interop.Memory
{
    /// <summary>
    /// Represents an enumeration for the different ways color channels can be ordered in a pixel of an image.
    /// </summary>
    public enum ChannelOrder : uint
    {
        /// <summary>
        /// Only a red channel.
        /// </summary>
        R = 0x10B0,

        /// <summary>
        /// Only an alpha channel.
        /// </summary>
        A = 0x10B1,

        /// <summary>
        /// Only a red and a green channel.
        /// </summary>
        Rg = 0x10B2,

        /// <summary>
        /// Only a red and an alpha channel.
        /// </summary>
        Ra = 0x10B3,

        /// <summary>
        /// Red, green, and blue channel. This channel order can only be used if the channel data type is <c>ChannelType.NormalizedUnsignedShortFloat565</c>, <c>ChannelType.NormalizedUnsignedShortFloat555</c> or
        /// <c>ChannelType.NormalizedUnsignedInteger101010</c>.
        /// </summary>
        Rgb = 0x10B4,

        /// <summary>
        /// Red, green, blue, and alpha channel.
        /// </summary>
        Rgba = 0x10B5,

        /// <summary>
        /// Blue, green, red, and alpha channel. This channel order can only be used if the channel data type is <c>ChannelType.NormalizedUnsignedInteger8</c>, <c>ChannelType.NormalizedSignedInteger8</c>, <c>ChannelType.SignedInteger8</c>, or
        /// <c>ChannelType.UnsignedInteger8</c>.
        /// </summary>
        Bgra = 0x10B6,

        /// <summary>
        /// Alpha, red, green, and blue channel. This channel order can only be used if the channel data type is <c>ChannelType.NormalizedUnsignedInteger8</c>, <c>ChannelType.NormalizedSignedInteger8</c>, <c>ChannelType.SignedInteger8</c>, or
        /// <c>ChannelType.UnsignedInteger8</c>.
        /// </summary>
        Argb = 0x10B7,

        /// <summary>
        /// Only one channel for the intensity. This channel order can only be used if the channel data type is <c>ChannelType.NormalizedUnsignedInteger8</c>, <c>ChannelType.NormalizedUnsignedInteger16</c>,
        /// <c>ChannelType.NormalizedSignedInteger8</c>, <c>ChannelType.NormalizedSignedInteger16</c>, <c>ChannelType.HalfFloat</c>, or <c>ChannelType.Float</c>.
        /// </summary>
        Intensity = 0x10B8,

        /// <summary>
        /// Only one channel for the luminance. This channel order can only be used if the channel data type is <c>ChannelType.NormalizedUnsignedInteger8</c>, <c>ChannelType.NormalizedUnsignedInteger16</c>,
        /// <c>ChannelType.NormalizedSignedInteger8</c>, <c>ChannelType.NormalizedSignedInteger16</c>, <c>ChannelType.HalfFloat</c>, or <c>ChannelType.Float</c>.
        /// </summary>
        Luminance = 0x10B9,

        /// <summary>
        /// Only a red channel with some padding.
        /// </summary>
        Rx = 0x10BA,

        /// <summary>
        /// Red and green channel with some padding.
        /// </summary>
        Rgx = 0x10BB,

        /// <summary>
        /// Red, green, and blue channel with some padding. This channel order can only be used if the channel data type is <c>ChannelType.NormalizedUnsignedShortFloat565</c>, <c>ChannelType.NormalizedUnsignedShortFloat555</c> or
        /// <c>ChannelType.NormalizedUnsignedInteger101010</c>.
        /// </summary>
        Rgbx = 0x10BC,

        /// <summary>
        /// Only one channel for depth. This channel order can only be used if the channel data type is <c>ChannelType.NormalizedUnsignedInteger16</c> or <c>ChannelType.Float</c>.
        /// </summary>
        Depth = 0x10BD,

        /// <summary>
        /// Only one channel for a depth stencil. This channel order can only be used if the channel data type is <c>ChannelType.NormalizedUnsignedInteger24</c>.
        /// </summary>
        DepthStencil = 0x10BE,

        /// <summary>
        /// Standard RGB (red, green, and blue channel). This channel order can only be used if the channel data type is <c>ChannelType.NormalizedUnsignedInteger8</c>.
        /// </summary>
        Srgb = 0x10BF,

        /// <summary>
        /// Standard RGB (red, green, and blue channel) with some padding. This channel order can only be used if the channel data type is <c>ChannelType.NormalizedUnsignedInteger8</c>.
        /// </summary>
        Srgbx = 0x10C0,

        /// <summary>
        /// Standard RGBA with alpha channel (red, green, blue, and alpha channel). This channel order can only be used if the channel data type is <c>ChannelType.NormalizedUnsignedInteger8</c>.
        /// </summary>
        Srgba = 0x10C1,

        /// <summary>
        /// Standard BGRA with alpha channel (blue, green, red, and alpha channel). This channel order can only be used if the channel data type is <c>ChannelType.NormalizedUnsignedInteger8</c>.
        /// </summary>
        Sbgra = 0x10C2,

        /// <summary>
        /// Alpha, blue, green, and red channel. This channel order can only be used if the channel data type is <c>ChannelType.NormalizedUnsignedInteger8</c>, <c>ChannelType.NormalizedSignedInteger8</c>, <c>ChannelType.SignedInteger8</c>, or
        /// <c>ChannelType.UnsignedInteger8</c>.
        /// </summary>
        Abgr = 0x10C3
    }
}