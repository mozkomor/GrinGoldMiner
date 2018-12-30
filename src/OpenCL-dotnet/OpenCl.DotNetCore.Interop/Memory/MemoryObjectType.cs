
namespace OpenCl.DotNetCore.Interop.Memory
{
    /// <summary>
    /// 
    /// </summary>
    public enum MemoryObjectType : uint
    {
        /// <summary>
        /// The memory object is a buffer.
        /// </summary>
        Buffer = 0x10F0,

        /// <summary>
        /// The memory object is a 2D image.
        /// </summary>
        Image2D = 0x10F1,

        /// <summary>
        /// The memory object is a 3D image.
        /// </summary>
        Image3D = 0x10F2,

        /// <summary>
        /// The memory object is a 2D image array.
        /// </summary>
        Image2DArray = 0x10F3,

        /// <summary>
        /// The memory object is a 1D image.
        /// </summary>
        Image1D = 0x10F4,

        /// <summary>
        /// The memory object is a 1D image array.
        /// </summary>
        Image1DArray = 0x10F5,

        /// <summary>
        /// The memory object is a 1D image buffer.
        /// </summary>
        Image1DBuffer = 0x10F6,

        /// <summary>
        /// The memory object is a pipe.
        /// </summary>
        Pipe = 0x10F7
    }
}