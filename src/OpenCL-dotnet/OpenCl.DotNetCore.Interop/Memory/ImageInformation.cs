
namespace OpenCl.DotNetCore.Interop.Memory
{
    /// <summary>
    /// Represents an enumeration for the different types of information that can be queried from an image object.
    /// </summary>
    public enum ImageInformation : uint
    {
        /// <summary>
        /// Return image format descriptor specified when image is created with <see cref="CreateImage"/>.
        /// </summary>
        Format = 0x1110,

        /// <summary>
        /// Return size of each element of the image memory object given by image in bytes. An element is made up of n channels. The value of n is given in image format descriptor.
        /// </summary>
        ElementSize = 0x1111,
        
        /// <summary>
        /// Return calculated row pitch in bytes of a row of elements of the image object given by image.
        /// </summary>
        RowPitch = 0x1112,
        
        /// <summary>
        /// Return calculated slice pitch in bytes of a 2D slice for the 3D image object or size of each image in a 1D or 2D image array given by image. For a 1D image, 1D image buffer and 2D image object return 0.
        /// </summary>
        SlicePitch = 0x1113,
        
        /// <summary>
        /// Return width of image in pixels.
        /// </summary>
        Width = 0x1114,
        
        /// <summary>
        /// Return height of image in pixels. For a 1D image, 1D image buffer and 1D image array object, height = 0.
        /// </summary>
        Height = 0x1115,
        
        /// <summary>
        /// Return depth of the image in pixels. For a 1D image, 1D image buffer, 2D image or 1D and 2D image array object, depth = 0.
        /// </summary>
        Depth = 0x1116,
        
        /// <summary>
        /// Return number of images in the image array. If image is not an image array, 0 is returned.
        /// </summary>
        ArraySize = 0x1117,
        
        /// <summary>
        /// Return buffer object associated with the image.
        /// </summary>
        Buffer = 0x1118,
        
        /// <summary>
        /// Return num_mip_levels associated with the image.
        /// </summary>
        NumberOfMipLevels = 0x1119,
        
        /// <summary>
        /// Return num_samples associated with the image.
        /// </summary>
        NumberOfSamples = 0x111A
    }
}