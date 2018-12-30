
#region Using Directives

using System;

#endregion

namespace OpenCl.DotNetCore.Interop.Devices
{
    /// <summary>
    /// Represents an enumeration that identifies the device information that can be queried from a device.
    /// </summary>
    public enum DeviceInformation : uint
    {
        /// <summary>
        /// The OpenCL device type.
        /// </summary>
        Type = 0x1000,

        /// <summary>
        /// A unique device vendor identifier. An example of a unique device identifier could be the PCIe ID.
        /// </summary>
        VendorId = 0x1001,

        /// <summary>
        /// The number of parallel compute units on the OpenCL device. A work-group executes on a single compute unit. The minimum value is 1.
        /// </summary>
        MaximumComputeUnits = 0x1002,

        /// <summary>
        /// Maximum dimensions that specify the global and local work-item IDs used by the data parallel execution model. (Refer to <see cref="EnqueueNDRangeKernel"/>). The minimum value is 3 for devices that are not of type
        /// <c>DeviceType.Custom</c>.
        /// </summary>
        MaximumWorkItemDimensions = 0x1003,

        /// <summary>
        /// Maximum number of work-items in a work-group that a device is capable of executing on a single compute unit, for any given kernel-instance running on the device. (Refer also to <see cref="EnqueueNDRangeKernel"/>) and
        /// <c>KernelWorkGroupInformation.WorkGroupSize</c>). The minimum value is 1.
        /// </summary>
        MaximumWorkGroupSize = 0x1004,

        /// <summary>
        /// Maximum number of work-items that can be specified in each dimension of the work-group to <see cref="EnqueueNDRangeKernel"/>. Retrieves n <see cref="UIntPtr"/> entries, where n is the value returned by the query for
        /// <c>DeviceInformation.MaximumWorkItemDimensions</c>. The minimum value is (1, 1, 1) for devices that are not of type <c>DeviceType.Custom</c>.
        /// </summary>
        MaximumWorkItemSizes = 0x1005,

        /// <summary>
        /// Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// </summary>
        PreferredVectorWidthChar = 0x1006,

        /// <summary>
        /// Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// </summary>
        PreferredVectorWidthShort = 0x1007,

        /// <summary>
        /// Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// </summary>
        PreferredVectorWidthInt = 0x1008,

        /// <summary>
        /// Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// </summary>
        PreferredVectorWidthLong = 0x1009,

        /// <summary>
        /// Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// </summary>
        PreferredVectorWidthFloat = 0x100A,

        /// <summary>
        /// Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector. If double precision is not supported,
        /// <c>DeviceInformation.PreferredVectorWidthDouble</c> must return 0.
        /// </summary>
        PreferredVectorWidthDouble = 0x100B,

        /// <summary>
        /// Maximum configured clock frequency of the device in MHz.
        /// </summary>
        MaximumClockFrequency = 0x100C,

        /// <summary>
        /// The default compute device address space size of the global address space specified as an unsigned integer value in bits. Currently supported values are 32 or 64 bits.
        /// </summary>
        AddressBits = 0x100D,

        /// <summary>
        /// Maximum number of image objects arguments of a kernel declared with the <c>read_only</c> qualifier. The minimum value is 128 if <c>DeviceInformation.ImageSupport</c> is <c>true</c>.
        /// </summary>
        MaximumReadImageArguments = 0x100E,

        /// <summary>
        /// Maximum number of image objects arguments of a kernel declared with the <c>write_only</c> qualifier. The minimum value is 64 if <c>DeviceInformation.ImageSupport</c> is <c>true</c>. Note:
        /// <c>DeviceInformation.MaximumWriteImageArguments</c> is only there for backward compatibility. <c>DeviceInformation.MaximumReadWriteImageArguments</c> should be used instead.
        /// </summary>
        MaximumWriteImageArguments = 0x100F,

        /// <summary>
        /// Maximum size of memory object allocation in bytes. The minimum value is max(min(1024*1024*1024, 1/4th of <c>DeviceInformation.GlobalMemorySize</c>), 32*1024*1024) for devices that are not of type <c>DeviceType.Custom</c>.
        /// </summary>
        MaximumMemoryAllocationSize = 0x1010,

        /// <summary>
        /// Maximum width of 2D image or 1D image not created from a buffer object in pixels. The minimum value is 16384 if <c>DeviceInformation.ImageSupport</c> is <c>true</c>.
        /// </summary>
        Image2DMaximumWidth = 0x1011,

        /// <summary>
        /// Maximum height of a 2D image in pixels. The minimum value is 16384 if <c>DeviceInformation.ImageSupport</c> is <c>true</c>.
        /// </summary>
        Image2DMaximumHeight = 0x1012,

        /// <summary>
        /// Maximum width of 3D image in pixels. The minimum value is 2048 if <c>DeviceInformation.ImageSupport</c> is <c>true</c>.
        /// </summary>
        Image3DMaximumWidth = 0x1013,

        /// <summary>
        /// Maximum height of 3D image in pixels. The minimum value is 2048 if <c>DeviceInformation.ImageSupport</c> is <c>true</c>.
        /// </summary>
        Image3DMaximumHeight = 0x1014,

        /// <summary>
        /// Maximum depth of 3D image in pixels. The minimum value is 2048 if <c>DeviceInformation.ImageSupport</c> is <c>true</c>.
        /// </summary>
        Image3DMaximumDepth = 0x1015,

        /// <summary>
        /// Is <c>true</c> if images are supported by the OpenCL device and <c>false</c> otherwise.
        /// </summary>
        ImageSupport = 0x1016,

        /// <summary>
        /// Maximum size in bytes of all arguments that can be passed to a kernel. The minimum value is 1024 for devices that are not of type <c>DeviceType.Custom</c>. For this minimum value, only a maximum of 128 arguments can be passed to
        /// a kernel.
        /// </summary>
        MaximumParameterSize = 0x1017,

        /// <summary>
        /// Maximum number of samplers that can be used in a kernel. The minimum value is 16 if <c>DeviceInformation.ImageSupport</c> is <c>true</c>.
        /// </summary>
        MaximumSamplers = 0x1018,

        /// <summary>
        /// Alignment requirement (in bits) for sub-buffer offsets. The minimum value is the size (in bits) of the largest OpenCL built-in data type supported by the device (long16 in the full profile, long16 or int16 in the embedded profile)
        /// for devices that are not of type <c>DeviceType.Custom</c>.
        /// </summary>
        MemoryBaseAddressAlignment = 0x1019,

        /// <summary>
        /// The smallest alignment in bytes which can be used for any data type.
        /// </summary>
        [Obsolete("MinimumDataTypeAlignmentSize is a deprecated OpenCL 1.1 property.")]
        MinimumDataTypeAlignmentSize = 0x101A,

        /// <summary>
        /// Describes single precision floating-point capability of the device. For the full profile, the mandated minimum floating-point capability for devices that are not of type <c>DeviceType.Custom</c> is
        /// <c>DeviceFloatingPointConfiguration.RoundToNearest | DeviceFloatingPointConfiguration.InfinityAndNotANumber</c>.
        /// </summary>
        SingleFloatingPointConfiguration = 0x101B,

        /// <summary>
        /// Type of global memory cache supported. Valid values are: <c>DeviceMemoryCacheType.None</c>, <c>DeviceMemoryCacheType.ReadOnlyCache</c>, and <c>DeviceMemoryCacheType.ReadWriteCache</c>.
        /// </summary>
        GlobalMemoryCacheType = 0x101C,

        /// <summary>
        /// Size of global memory cache line in bytes.
        /// </summary>
        GlobalMemoryCachelineSize = 0x101D,

        /// <summary>
        /// Size of global memory cache in bytes.
        /// </summary>
        GlobalMemoryCacheSize = 0x101E,

        /// <summary>
        /// Size of global device memory in bytes.
        /// </summary>
        GlobalMemorySize = 0x101F,

        /// <summary>
        /// Maximum size in bytes of a constant buffer allocation. The minimum value is 64 KB for devices that are not of type <c>DeviceType.Custom</c>.
        /// </summary>
        MaximumConstantBufferSize = 0x1020,

        /// <summary>
        /// Maximum number of arguments declared with the <c>__constant</c> qualifier in a kernel. The minimum value is 8 for devices that are not of type <c>DeviceType.Custom</c>.
        /// </summary>
        MaximumConstantArguments = 0x1021,

        /// <summary>
        /// Type of local memory supported. This can be set to <c>DeviceLocalMemoryType.Local</c> implying dedicated local memory storage such as SRAM, or <c>DeviceLocalMemoryType.Global</c>. For custom devices,
        /// <c>DeviceLocalMemoryType.None</c> can also be returned indicating no local memory support.
        /// </summary>
        LocalMemoryType = 0x1022,

        /// <summary>
        /// Size of local memory region in bytes. The minimum value is 32 KB for devices that are not of type <c>DeviceType.Custom</c>.
        /// </summary>
        LocalMemorySize = 0x1023,

        /// <summary>
        /// Is <c>true</c> if the device implements error correction for all accesses to compute device memory (global and constant). Is <c>false</c> if the device does not implement such error correction.
        /// </summary>
        ErrorCorrectionSupport = 0x1024,

        /// <summary>
        /// Describes the resolution of device timer. This is measured in nanoseconds.
        /// </summary>
        ProfilingTimerResolution = 0x1025,

        /// <summary>
        /// Is <c>true</c> if the OpenCL device is a little endian device and <c>false</c> otherwise.
        /// </summary>
        EndianLittle = 0x1026,

        /// <summary>
        /// Is <c>true</c> if the device is available and <c>false</c> otherwise. A device is considered to be available if the device can be expected to successfully execute commands enqueued to the .
        /// </summary>
        Available = 0x1027,

        /// <summary>
        /// Is <c>false</c> if the implementation does not have a compiler available to compile the program source. Is <c>true</c> if the compiler is available. This can be <c>false</c> for the embedded platform profile only.
        /// </summary>
        CompilerAvailable = 0x1028,

        /// <summary>
        /// Describes the execution capabilities of the device. The mandated minimum capability is <c>DeviceExecutionCapability.Kernel</c>.
        /// </summary>
        ExecutionCapabilities = 0x1029,

        /// <summary>
        /// Describes the command-queue properties supported by the device. The mandated minimum capability is <c>CommandQueueProperty.ProfilingEnable</c>.
        /// </summary>
        [Obsolete("QueueProperties is a deprecated OpenCL 1.2 property, please use QueueOnHostProperties.")]
        QueueProperties = 0x102A,

        /// <summary>
        /// Describes the on host command-queue properties supported by the device. The mandated minimum capability is <c>CommandQueueProperty.ProfilingEnable</c>.
        /// </summary>
        QueueOnHostProperties = 0x102A,

        /// <summary>
        /// Device name string.
        /// </summary>
        Name = 0x102B,

        /// <summary>
        /// Vendor name string.
        /// </summary>
        Vendor = 0x102C,

        /// <summary>
        /// OpenCL software driver version string in the form major_number.minor_number.
        /// </summary>
        DriverVersion = 0x102D,

        /// <summary>
        /// OpenCL profile string. Retrieves the profile name supported by the device (see note). The profile name returned can be one of the following strings:
        /// 
        /// FULL_PROFILE - if the device supports the OpenCL specification (functionality defined as part of the core specification and does not require any extensions to be supported).
        /// 
        /// EMBEDDED_PROFILE - if the device supports the OpenCL embedded profile.
        /// 
        /// The platform profile returns the profile that is implemented by the OpenCL framework. If the platform profile returned is FULL_PROFILE, the OpenCL framework will support devices that are FULL_PROFILE and may also support devices
        /// that are EMBEDDED_PROFILE. The compiler must be available for all devices i.e. <c>DeviceInformation.CompilerAvailable</c> is <c>true</c>. If the platform profile returned is EMBEDDED_PROFILE, then devices that are only
        /// EMBEDDED_PROFILE are supported.
        /// </summary>
        Profile = 0x102E,

        /// <summary>
        /// OpenCL version string. Returns the OpenCL version supported by the device. This version string has the following format:
        /// 
        /// OpenCL[space][major_version.minor_version][space][vendor-specific information]
        /// </summary>
        Version = 0x102F,

        /// <summary>
        /// Retrieves a space separated list of extension names (the extension names themselves do not contain any spaces) supported by the device. The list of extension names returned can be vendor supported extension names and one or more of
        /// the following Khronos approved extension names:
        /// 
        /// cl_khr_int64_base_atomics, cl_khr_int64_extended_atomics, cl_khr_fp16, cl_khr_gl_sharing, cl_khr_gl_event, cl_khr_d3d10_sharing, cl_khr_dx9_media_sharing, cl_khr_d3d11_sharing, cl_khr_gl_depth_images, cl_khr_gl_msaa_sharing,
        /// cl_khr_initialize_memory, cl_khr_context_abort, cl_khr_spir, and cl_khr_srgb_image_writes.
        /// 
        /// The following approved Khronos extension names must be returned by all device that support OpenCL C 2.0:
        /// 
        /// cl_khr_byte_addressable_store, cl_khr_fp64 (for backward compatibility if double precision is supported), cl_khr_3d_image_writes, cl_khr_image2d_from_buffer, and cl_khr_depth_images.
        /// 
        /// Please refer to the OpenCL 2.0 Extension Specification for a detailed description of these extensions.
        /// </summary>
        Extensions = 0x1030,

        /// <summary>
        /// The platform associated with this device.
        /// </summary>
        Platform = 0x1031,

        /// <summary>
        /// Describes double precision floating-point capability of the OpenCL device. Double precision is an optional feature so the mandated minimum double precision floating-point capability is 0. If double precision is supported by the
        /// device, then the minimum double precision floating-point capability must be:
        /// <c>DeviceFloatingPointConfiguration.FusedMultiplyAdd | DeviceFloatingPointConfiguration.RoundToNearest | DeviceFloatingPointConfiguration.InfinityAndNotANumber | DeviceFloatingPointConfiguration.Denorm</c>.
        /// </summary>
        DoubleFloatingPointConfiguration = 0x1032,

        /// <summary>
        /// Describes the optional half precision floating-point capability of the OpenCL device. The required minimum half precision floating-point capability as implemented by this extension is
        /// <c>DeviceFloatingPointConfiguration.RoundToZero</c> or <c>DeviceFloatingPointConfiguration.RoundToInfinity | DeviceFloatingPointConfiguration.InfinityAndNotANumber</c>.
        /// </summary>
        HalfFloatingPointConfiguration = 0x1033,

        /// <summary>
        /// Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector. If the cl_khr_fp16 extension is not supported,
        /// <c>DeviceInformation.PreferredVectorWidthHalf</c> must return 0.
        /// </summary>
        PreferredVectorWidthHalf = 0x1034,

        /// <summary>
        /// Is <c>true</c> if the device and the host have a unified memory subsystem and is <c>false</c> otherwise.
        /// </summary>
        [Obsolete("HostUnifiedMemory is a deprecated OpenCL 1.2 property.")]
        HostUnifiedMemory = 0x1035,

        /// <summary>
        /// Retrieves the native ISA vector width. The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// </summary>
        NativeVectorWidthChar = 0x1036,

        /// <summary>
        /// Retrieves the native ISA vector width. The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// </summary>
        NativeVectorWidthShort = 0x1037,

        /// <summary>
        /// Retrieves the native ISA vector width. The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// </summary>
        NativeVectorWidthInt = 0x1038,

        /// <summary>
        /// Retrieves the native ISA vector width. The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// </summary>
        NativeVectorWidthLong = 0x1039,

        /// <summary>
        /// Retrieves the native ISA vector width. The vector width is defined as the number of scalar elements that can be stored in the vector.
        /// </summary>
        NativeVectorWidthFloat = 0x103A,

        /// <summary>
        /// Retrieves the native ISA vector width. The vector width is defined as the number of scalar elements that can be stored in the vector. If double precision is not supported, <c>DeviceInformation.NativeVectorWidthDouble</c> must
        /// return 0.
        /// </summary>
        NativeVectorWidthDouble = 0x103B,

        /// <summary>
        /// Retrieves the native ISA vector width. The vector width is defined as the number of scalar elements that can be stored in the vector. If the cl_khr_fp16 extension is not supported, <c>DeviceInformation.NativeVectorWidthHalf</c>
        /// must return 0.
        /// </summary>
        NativeVectorWidthHalf = 0x103C,

        /// <summary>
        /// OpenCL C version string. Retrieves the highest OpenCL C version supported by the compiler for this device that is not of type <c>DeviceType.Custom</c>. This version string has the following format:
        /// 
        /// OpenCL[space]C[space][major_version.minor_version][space][vendor-specific information]
        /// The major_version.minor_version value returned must be 2.0 if <c>DeviceInformation.Version</c> is OpenCL 2.0.
        /// The major_version.minor_version value returned must be 1.2 if <c>DeviceInformation.Version</c> is OpenCL 1.2.
        /// The major_version.minor_version value returned must be 1.1 if <c>DeviceInformation.Version</c> is OpenCL 1.1.
        /// The major_version.minor_version value returned can be 1.0 or 1.1 if <c>DeviceInformation.Version</c> is OpenCL 1.0.
        /// </summary>
        OpenClCVersion = 0x103D,

        /// <summary>
        /// Is <c>false</c> if the implementation does not have a linker available. Is <c>true</c> if the linker is available. This can be <c>false</c> for the embedded platform profile only. This must be <c>true</c> if
        /// <c>DeviceInformation.CompilerAvailable</c> is <c>true</c>.
        /// </summary>
        LinkerAvailable = 0x103E,

        /// <summary>
        /// A semi-colon separated list of built-in kernels supported by the device. An empty string is returned if no built-in kernels are supported by the device.
        /// </summary>
        BuiltInKernels = 0x103F,

        /// <summary>
        /// Maximum number of pixels for a 1D image created from a buffer object. The minimum value is 65536 if <c>DeviceInformation.ImageSupport</c> is <c>true</c>.
        /// </summary>
        ImageMaximumBufferSize = 0x1040,

        /// <summary>
        /// Maximum number of images in a 1D or 2D image array. The minimum value is 2048 if <c>DeviceInformation.ImageSupport</c> is <c>true</c>.
        /// </summary>
        ImageMaximumArraySize = 0x1041,

        /// <summary>
        /// Retrieves the handle of the parent device to which this sub-device belongs. If device is a root-level device, a <c>null</c> value is returned.
        /// </summary>
        ParentDevice = 0x1042,

        /// <summary>
        /// Retrieves the maximum number of sub-devices that can be created when a device is partitioned. The value returned cannot exceed <c>DeviceInformation.MaximumComputeUnits</c>.
        /// </summary>
        PartitionMaximumSubDevices = 0x1043,

        /// <summary>
        /// Retrieves the list of partition types supported by device. If the device cannot be partitioned (i.e. there is no partitioning scheme supported by the device that will return at least two subdevices), a value of 0 will be returned.
        /// </summary>
        PartitionProperties = 0x1044,

        /// <summary>
        /// Retrieves the list of supported affinity domains for partitioning the device using <c>DevicePartitionProperty.PartitionByAffinityDomain</c>. If the device does not support any affinity domains, a value of 0 will be returned.
        /// </summary>
        PartitionAffinityDomain = 0x1045,

        /// <summary>
        /// Retrieves the properties argument specified in <see cref="CreateSubDevices"/> if the device is a sub-device. In the case where the properties argument to <see cref="CreateSubDevices"/> is
        /// <c>DevicePartitionProperty.PartitionByAffinityDomain</c>, <c>DeviceAffinityDomain.NextPartitionable</c>, the affinity domain used to perform the partition will be returned. Otherwise the implementation may either return a
        /// <see cref="parameterValueSize"/> of 0 i.e. there is no partition type associated with device or can return a property value of 0 (where 0 is used to terminate the partition property list) in the memory that
        /// <see cref="parameterValue"/> points to.
        /// </summary>
        PartitionType = 0x1046,

        /// <summary>
        /// Retrieves the device reference count. If the device is a root-level device, a reference count of one is returned.
        /// </summary>
        ReferenceCount = 0x1047,

        /// <summary>
        /// Is <c>true</c> if the device's preference is for the user to be responsible for synchronization, when sharing memory objects between OpenCL and other APIs such as DirectX, <c>false</c> if the device/implementation has a
        /// performant path for performing synchronization of memory object shared between OpenCL and other APIs such as DirectX.
        /// </summary>
        PreferredInteropUserSync = 0x1048,

        /// <summary>
        /// Maximum size in bytes of the internal buffer that holds the output of printf calls from a kernel. The minimum value for the full profile is 1 MB.
        /// </summary>
        PrintfBufferSize = 0x1049,

        /// <summary>
        /// The row pitch alignment size in pixels for 2D images created from a buffer. The value returned must be a power of 2. If the device does not support images, this value must be 0.
        /// </summary>
        ImagePitchAlignment = 0x104A,

        /// <summary>
        /// This query should be used when a 2D image is created from a buffer which was created using <c>MemoryFlag.UseHostPointer</c>. The value returned must be a power of 2. This query specifies the minimum alignment in pixels of the
        /// <see cref="hostPointer"/> specified to <see cref="CreateBuffer"/>. If the device does not support images, this value must be 0.
        /// </summary>
        ImageBaseAddressAlignment = 0x104B,

        /// <summary>
        /// Maximum number of image objects arguments of a kernel declared with the <c>write_only</c> or <c>read_write</c> qualifier. The minimum value is 64 if <c>DeviceInformation.ImageSupport</c> is <c>true</c>. Note:
        /// <c>DeviceInformation.MaximumWriteImageArguments</c> is only there for backward compatibility. <c>DeviceInformation.MaximumReadWriteImageArguments</c> should be used instead.
        /// </summary>
        MaximumReadWriteImageArguments = 0x104C,

        /// <summary>
        /// The maximum number of bytes of storage that may be allocated for any single variable in program scope or inside a function in OpenCL C declared in the global address space. The minimum value is 64 KB.
        /// </summary>
        MaximumGlobalVariableSize = 0x104D,

        /// <summary>
        /// Describes the on device command-queue properties supported by the device. The mandated minimum capability is <c>CommandQueueProperty.OutOfOrderExecutionModeEnable | CommandQueueProperty.ProfilingEnable</c>.
        /// </summary>
        QueueOnDeviceProperties = 0x104E,

        /// <summary>
        /// The size of the device queue in bytes preferred by the implementation. Applications should use this size for the device queue to ensure good performance. The minimum value is 16 KB.
        /// </summary>
        QueueOnDevicePreferredSize = 0x104F,

        /// <summary>
        /// The maximum size of the device queue in bytes. The minimum value is 256 KB for the full profile and 64 KB for the embedded profile.
        /// </summary>
        QueueOnDeviceMaximumSize = 0x1050,

        /// <summary>
        /// The maximum number of device queues that can be created per context. The minimum value is 1.
        /// </summary>
        MaximumOnDeviceQueues = 0x1051,

        /// <summary>
        /// The maximum number of events in use by a device queue. These refer to events returned by the <c>enqueue_</c> built-in functions to a device queue or user events returned by the <c>create_user_event</c> built-in function that have
        /// not been released. The minimum value is 1024.
        /// </summary>
        MaximumOnDeviceEvents = 0x1052,

        /// <summary>
        /// Describes the various shared virtual memory (a.k.a. SVM) memory allocation types the device supports. Coarse-grain SVM allocations are required to be supported by all OpenCL 2.0 devices. The mandated minimum capability is
        /// <c>DeviceSvmCapabilities.CoarseGrainBuffer</c>.
        /// </summary>
        SvmCapabilities = 0x1053,

        /// <summary>
        /// Maximum preferred total size, in bytes, of all program variables in the global address space. This is a performance hint. An implementation may place such variables in storage with optimized device access. This query returns the
        /// capacity of such storage. The minimum value is 0.
        /// </summary>
        GlobalVariablePreferredTotalSize = 0x1054,

        /// <summary>
        /// The maximum number of pipe objects that can be passed as arguments to a kernel. The minimum value is 16.
        /// </summary>
        MaximumPipeArguments = 0x1055,

        /// <summary>
        /// The maximum number of reservations that can be active for a pipe per work-item in a kernel. A work-group reservation is counted as one reservation per work-item. The minimum value is 1.
        /// </summary>
        PipeMaximumActiveReservations = 0x1056,

        /// <summary>
        /// The maximum size of pipe packet in bytes. The minimum value is 1024 bytes.
        /// </summary>
        PipeMaximumPacketSize = 0x1057,

        /// <summary>
        /// Retrieves the value representing the preferred alignment in bytes for OpenCL 2.0 fine-grained SVM atomic types. This query can return 0 which indicates that the preferred alignment is aligned to the natural size of the type.
        /// </summary>
        PreferredPlatformAtomicAlignment = 0x1058,

        /// <summary>
        /// Retrieves the value representing the preferred alignment in bytes for OpenCL 2.0 atomic types to global memory. This query can return 0 which indicates that the preferred alignment is aligned to the natural size of the type.
        /// </summary>
        PreferredGlobalAtomicAlignment = 0x1059,

        /// <summary>
        /// Retrieves the value representing the preferred alignment in bytes for OpenCL 2.0 atomic types to local memory. This query can return 0 which indicates that the preferred alignment is aligned to the natural size of the type.
        /// </summary>
        PreferredLocalAtomicAlignment = 0x105A,

        /// <summary>
        /// The intermediate languages that can be supported by <see cref="CreateProgramWithIL"/> for this device. Retrieves a space-separated list of IL version strings of the form <IL_Prefix>_<Major_Version>.<Minor_Version>. For OpenCL 2.1,
        /// "SPIR-V" is a required IL prefix.
        /// </summary>
        IntermediateLanguageVersion = 0x105B,

        /// <summary>
        /// Maximum number of sub-groups in a work-group that a device is capable of executing on a single compute unit, for any given kernel-instance running on the device. The minimum value is 1. (Refer also to
        /// <see cref="GetKernelSubGroupInfo"/>.)
        /// </summary>
        MaximumNumberOfSubGroups = 0x105C,

        /// <summary>
        /// Is <c>true</c> if this device supports independent forward progress of sub-groups, <c>false</c> otherwise. If cl_khr_subgroups is supported by the device this must return <c>true</c>.
        /// </summary>
        SubGroupIndependentForwardProgress = 0x105D,

        /// <summary>
        /// If the cl_khr_terminate_context extension is enabled, describes the termination capability of the OpenCL device. This is a bitfield where a value of <c>TerminateCapabilityContextKhr</c> indicates that context termination is
        /// supported.
        /// </summary>
        TerminateCapabilityKhr = 0x2031,

        /// <summary>
        /// If the cl_khr_spir extension is enabled, a space separated list of SPIR versions supported by the device. For example returning “1.2 2.0” in this query implies that SPIR version 1.2 and 2.0 are supported by the implementation.
        /// </summary>
        SpirVersion = 0x40E0
    }
}