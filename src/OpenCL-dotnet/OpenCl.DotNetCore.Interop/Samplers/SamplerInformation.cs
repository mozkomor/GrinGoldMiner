
namespace OpenCl.DotNetCore.Interop.Samplers
{
    /// <summary>
    /// Represents an enumeration for the different types of information that can be queried from a sampler object.
    /// </summary>
    public enum SamplerInformation : uint
    {
        /// <summary>
        /// Retrieves the sampler reference count. The reference count returned should be considered immediately stale. It is unsuitable for general use in applications. This feature is provided for identifying memory leaks.
        /// </summary>
        ReferenceCount = 0x1150,

        /// <summary>
        /// Retrieves the context specified when the sampler is created.
        /// </summary>
        Context = 0x1151,
        
        /// <summary>
        /// Retrieves the normalized coordinates value associated with the sampler.
        /// </summary>
        NormalizedCoordinates = 0x1152,
        
        /// <summary>
        /// Retrieves the addressing mode value associated with the sampler.
        /// </summary>
        AddressingMode = 0x1153,
        
        /// <summary>
        /// Retrieves the filter mode value associated with the sampler.
        /// </summary>
        FilterMode = 0x1154,
        
        /// <summary>
        /// Retrievess the MIP filter mode.
        /// </summary>
        MipFilterMode = 0x1155,
        
        /// <summary>
        /// Retrieves the LOD minimum.
        /// </summary>
        LodMinimum = 0x1156,
        
        /// <summary>
        /// Retrieves the LOD maximum.
        /// </summary>
        LodMaximum = 0x1157
    }
}