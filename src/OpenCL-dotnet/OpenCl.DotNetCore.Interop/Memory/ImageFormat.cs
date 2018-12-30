
#region Using Directives

using System.Runtime.InteropServices;

#endregion

namespace OpenCl.DotNetCore.Interop.Memory
{
    /// <summary>
    /// Represents the image format descriptor structure.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ImageFormat
    {
        #region Public Properties

        /// <summary>
        /// Contains a value that specifies the number of channels and the channel layout i.e. the memory layout in which channels are stored in the image.
        /// </summary>
        public ChannelOrder ChannelOrder;

        /// <summary>
        /// Contains a value that describes the size of the channel data type. The number of bits per element determined by the <see cref="ChannelDataType"/> and <see cref="ChannelOrder"/> must be a power of two.
        /// </summary>
        public ChannelType ChannelDataType;

        #endregion
    }
}