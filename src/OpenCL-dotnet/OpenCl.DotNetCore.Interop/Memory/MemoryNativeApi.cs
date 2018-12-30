
#region Using Directives

using System;
using System.Runtime.InteropServices;

#endregion

namespace OpenCl.DotNetCore.Interop.Memory
{
    /// <summary>
    /// Represents a wrapper for the native methods of the OpenCL Memory API.
    /// </summary>
    public static class MemoryNativeApi
    {
        #region Public Static Methods

        /// <summary>
        /// Creates a buffer object.
        /// </summary>
        /// <param name="context">A valid OpenCL context used to create the buffer object.</param>
        /// <param name="flags">
        /// An enumeration that is used to specify allocation and usage information such as the memory arena that should be used to allocate the buffer object and how it will be used. If value specified for <see cref="flags"/> is 0, the
        /// default is used which is <see cref="MemoryFlag.ReadWrite"/>.
        /// </param>
        /// <param name="size">The size in bytes of the buffer memory object to be allocated.</param>
        /// <param name="hostPointer">A pointer to the buffer data that may already be allocated by the application. The size of the buffer that <see cref="hostPointer"/> points to must be greater or equal than size bytes.</param>
        /// <param name="errorCode">Returns an appropriate error code. If <see cref="errorCode"/> is <c>null</c>, no error code is returned.</param>
        /// <returns>
        /// Returns a valid non-zero buffer object and <see cref="errorCode"/> is set to <c>Result.Success</c> if the buffer object is created successfully. Otherwise, it returns a <c>null</c> value and an error value in
        /// <see cref="errorCode"/>.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clCreateBuffer")]
        public static extern IntPtr CreateBuffer(
            [In] IntPtr context,
            [In] [MarshalAs(UnmanagedType.U8)] MemoryFlag flags,
            [In] UIntPtr size,
            [In] IntPtr hostPointer,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        /// <summary>
        /// Creates a new buffer object (referred to as a sub-buffer object) from an existing buffer object.
        /// </summary>
        /// <param name="memoryObject">A valid buffer object and cannot be a sub-buffer object.</param>
        /// <param name="flags">
        /// An enumeration that is used to specify allocation and usage information about the sub-buffer memory object being created and is described in the table below. If the <c>MemoryFlag.ReadWrite</c>, <c>MemoryFlag.ReadOnly</c> or
        /// <c>MemoryFlag.WriteOnly</c> values are not specified in <see cref="flags"/>, they are inherited from the corresponding memory access qualifers associated with <see cref="buffer"/>. The <c>MemoryFlag.UseHostPointer</c>,
        /// <c>MemoryFlag.AllocateHostPointer</c> and <c>MemoryFlag.CopyHostPointer</c> values cannot be specified in <see cref="flags"/> but are inherited from the corresponding memory access qualifiers associated with <see cref="buffer"/>.
        /// If <c>MemoryFlag.CopyHostPointer</c> is specified in the memory access qualifier values associated with <see cref="buffer"/> it does not imply any additional copies when the sub-buffer is created from <see cref="buffer"/>. If
        /// the <c>MemoryFlag.HostWriteOnly</c>, <c>MemoryFlag.HostReadOnly</c> or <c>MemoryFlag.HostNoAccess</c> values are not specified in <see cref="flags"/>, they are inherited from the corresponding memory access qualifiers associated
        /// with <see cref="buffer"/>.
        /// </param>
        /// <param name="bufferCreateType">Describes the type of buffer object to be created.</param>
        /// <param name="bufferCreateInformation">Describes the type of buffer object to be created. </param>
        /// <param name="errorCode">Returns an appropriate error code. If <see cref="errorCode"/> is <c>null</c>, no error code is returned.</param>
        /// <returns>
        /// Returns a valid non-zero buffer object and <see cref="errorCode"/> is set to <c>Result.Success</c> if the buffer object is created successfully. Otherwise, it returns a <c>null</c> value and an error value in
        /// <see cref="errorCode"/>.
        /// </returns>
        [IntroducedInOpenCl(1, 1)]
        [DllImport("OpenCL", EntryPoint = "clCreateSubBuffer")]
        public static extern IntPtr CreateSubBuffer(
            [In] IntPtr memoryObject,
            [In] [MarshalAs(UnmanagedType.U8)] MemoryFlag flags,
            [In] [MarshalAs(UnmanagedType.U4)] BufferCreateType bufferCreateType,
            [In] IntPtr bufferCreateInformation,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        /// <summary>
        /// Creates a 1D image, 1D image buffer, 1D image array, 2D image, 2D image array or 3D image object.
        /// </summary>
        /// <param name="context">A valid OpenCL context on which the image object is to be created.</param>
        /// <param name="flags">An enumeration that is used to specify allocation and usage information about the image memory object being created.</param>
        /// <param name="imageFormat">A pointer to a structure that describes format properties of the image to be allocated.</param>
        /// <param name="imageDescription">A pointer to a structure that describes type and dimensions of the image to be allocated.</param>
        /// <param name="hostPointer">A pointer to the image data that may already be allocated by the application.</param>
        /// <param name="errorCode">Returns an appropriate error code. If <see cref="errorCode"/> is <c>null</c>, no error code is returned.</param>
        /// <returns>
        /// Returns a valid non-zero image object and <see cref="errorCode"/> is set to <c>Result.Success</c> if the image object is created successfully. Otherwise, it returns a <c>null</c> value with an error value returned in
        /// <see cref="errorCode"/>.
        /// </returns>
        [IntroducedInOpenCl(1, 2)]
        [DllImport("OpenCL", EntryPoint = "clCreateImage")]
        public static extern IntPtr CreateImage(
            [In] IntPtr context,
            [In] [MarshalAs(UnmanagedType.U8)] MemoryFlag flags,
            [In] IntPtr imageFormat,
            [In] IntPtr imageDescription,
            [In] IntPtr hostPointer,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        /// <summary>
        /// Creates a pipe object.
        /// </summary>
        /// <param name="context">A valid OpenCL context used to create the pipe object.</param>
        /// <param name="flags">
        /// An enumeration that is used to specify allocation and usage information such as the memory arena that should be used to allocate the pipe object and how it will be used. Only <c>MemoryFlag.ReadOnly</c>, <c>MemoryFlag.WriteOnly</c>,
        /// <c>MemoryFlag.ReadWrite</c>, and <c>MemoryFlag.HostNoAccess</c> can be specified when creating a pipe object. If value specified for <see cref="flags"/> is 0, the default is used which is
        /// <c>MemoryFlag.ReadWrite | MemoryFlag.HostNoAccess</c>.
        /// </param>
        /// <param name="pipePacketSize">Size in bytes of a pipe packet.</param>
        /// <param name="pipeMaximumNumberOfPackets">Specifies the pipe capacity by specifying the maximum number of packets the pipe can hold.</param>
        /// <param name="properties">
        /// A list of properties for the pipe and their corresponding values. Each property name is immediately followed by the corresponding desired value. The list is terminated with 0. In OpenCL 2.0, properties must be <c>null</c>.
        /// </param>
        /// <param name="errorCode">Returns an appropriate error code. If <see cref="errorCode"/> is <c>null</c>, no error code is returned.</param>
        /// <returns>
        /// Returns a valid non-zero pipe object and <see cref="errorCode"/> is set to <c>Result.Success</c> if the pipe object is created successfully. Otherwise, it returns a <c>null</c> value with an error value returned in
        /// <see cref="errorCode"/>.
        /// </returns>
        [IntroducedInOpenCl(2, 0)]
        [DllImport("OpenCL", EntryPoint = "clCreatePipe")]
        public static extern IntPtr CreatePipe(
            [In] IntPtr context,
            [In] [MarshalAs(UnmanagedType.U8)] MemoryFlag flags,
            [In] [MarshalAs(UnmanagedType.U4)] uint pipePacketSize,
            [In] [MarshalAs(UnmanagedType.U4)] uint pipeMaximumNumberOfPackets,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] properties,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        /// <summary>
        /// Increments the memory object reference count.
        /// </summary>
        /// <param name="memoryObject">A valid memory object.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidMemoryObject</c> if <see cref="memoryObject"/> is not a valid memory object.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clRetainMemObject")]
        public static extern Result RetainMemoryObject(
            [In] IntPtr memoryObject
        );

        /// <summary>
        /// Decrements the memory object reference count.
        /// </summary>
        /// <param name="memoryObject">Specifies the memory object to release.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidContext</c> if <see cref="memoryObject"/> is not a valid memory object.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clReleaseMemObject")]
        public static extern Result ReleaseMemoryObject(
            [In] IntPtr memoryObject
        );

        /// <summary>
        /// Get the list of image formats supported by an OpenCL implementation.
        /// </summary>
        /// <param name="context">A valid OpenCL context on which the image object(s) will be created.</param>
        /// <param name="flags">An enumeration that is used to specify allocation and usage information about the image memory object being queried.</param>
        /// <param name="imageType">Describes the image type.</param>
        /// <param name="numberOfEntries">Specifies the number of entries that can be returned in the memory location given by <see cref="imageFormats"/>.</param>
        /// <param name="imageFormats">
        /// A pointer to a memory location where the list of supported image formats are returned. Each entry describes a <see cref="ImageFormat"/> structure supported by the OpenCL implementation. If image_formats is <c>null</c>, it is ignored.
        /// </param>
        /// <param name="numberOfImageFormats">The actual number of supported image formats for a specific <see cref="context"/> and values specified by <see cref="flags"/>. If <see cref="numberOfImageFormats"/> is <c>null</c>, it is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following:
        /// 
        /// <c>Result.InvalidContext</c> if <see cref="context"/> is not a valid context.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="flags"/> or <see cref="imageType"/> are not valid or if <see cref="numberOfEntries"/> is 0 and <see cref="imageFormats"/> is not <c>null</c>.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clGetSupportedImageFormats")]
        public static extern Result GetSupportedImageFormats(
            [In] IntPtr context,
            [In] [MarshalAs(UnmanagedType.U8)] MemoryFlag flags,
            [In] [MarshalAs(UnmanagedType.U4)] MemoryObjectType imageType,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEntries,
            [Out] [MarshalAs(UnmanagedType.LPArray)] ImageFormat[] imageFormats,
            [Out] [MarshalAs(UnmanagedType.U4)] out uint numberOfImageFormats
        );

        /// <summary>
        /// Get information that is common to all memory objects (buffer and image objects).
        /// </summary>
        /// <param name="memoryObject">Specifies the memory object being queried.</param>
        /// <param name="parameterName">Specifies the information to query.</param>
        /// <param name="parameterValueSize">Used to specify the size in bytes of memory pointed to by <see cref="parameterValue"/>. This size must be greater or equal to the size of the return type.</param>
        /// <param name="parameterValue">A pointer to memory where the appropriate result being queried is returned. If <see cref="parameterValue"/> is <c>null</c>, it is ignored.</param>
        /// <param name="parameterValueSizeReturned">The actual size in bytes of data copied to <see cref="parameterValue"/>. If <see cref="parameterValueSizeReturned"/> is <c>null</c>, it is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following:
        /// 
        /// <c>Result.InvalidMemoryObject</c> if <see cref="memoryObject"/> is not a valid memory object.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="parameterName"/> is not one of the supported values or if size in bytes specified by <see cref="parameterValueSize"/> is less than size of return type and <see cref="parameterValue"/>
        /// is not a <c>null</c> value or if <see cref="parameterName"/> is a value that is available as an extension and the corresponding extension is not supported by the device.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clGetMemObjectInfo")]
        public static extern Result GetMemoryObjectInformation(
            [In] IntPtr memoryObject,
            [In] [MarshalAs(UnmanagedType.U4)] MemoryObjectInformation parameterName,
            [In] UIntPtr parameterValueSize,
            [Out] byte[] parameterValue,
            [Out] out UIntPtr parameterValueSizeReturned
        );

        /// <summary>
        /// Get information specific to an image object created with <see cref="CreateImage"/>.
        /// </summary>
        /// <param name="image">Specifies the image object being queried.</param>
        /// <param name="parameterName">Specifies the information to query.</param>
        /// <param name="parameterValueSize">Used to specify the size in bytes of memory pointed to by <see cref="parameterValue"/>. This size must be greater or equla to the size of the return type.</param>
        /// <param name="parameterValue">A pointer to memory where the appropriate result being queried is returned. If <see cref="parameterValue"/> is <c>null</c>, it is ignored.</param>
        /// <param name="parameterValueSizeReturned">Returns the actual size in bytes of data being queried by <see cref="parameterValue"/>. If <see cref="parameterValueSizeReturned"/> is <c>null</c>, it is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following:
        /// 
        /// <c>Result.InvalidMemoryObject</c> if <see cref="image"/> is not a valid image object.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="parameterName"/> is not valid, or if size in bytes specified by <see cref="parameterValueSize"/> is less than the size of return type as described in the table above and
        /// <see cref="parameterValue"/> is not <c>null</c>.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clGetImageInfo")]
        public static extern Result GetImageInformation(
            [In] IntPtr image,
            [In] [MarshalAs(UnmanagedType.U4)] ImageInformation parameterName,
            [In] UIntPtr parameterValueSize,
            [Out] byte[] parameterValue,
            [Out] out UIntPtr parameterValueSizeReturned
        );

        /// <summary>
        /// Get information specific to a pipe object created with <see cref="CreatePipe"/>.
        /// </summary>
        /// <param name="pipe">Specifies the pipe object being queried.</param>
        /// <param name="parameterName">Specifies the information to query.</param>
        /// <param name="parameterValueSize">Used to specify the size in bytes of memory pointed to by <see cref="parameterValue"/>. This size must be greater or equla to the size of the return type.</param>
        /// <param name="parameterValue">A pointer to memory where the appropriate result being queried is returned. If <see cref="parameterValue"/> is <c>null</c>, it is ignored.</param>
        /// <param name="parameterValueSizeReturned">Returns the actual size in bytes of data being queried by <see cref="parameterValue"/>. If <see cref="parameterValueSizeReturned"/> is <c>null</c>, it is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following:
        /// 
        /// <c>Result.InvalidMemoryObject</c> if <see cref="pipe"/> is not a valid pipe object.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="parameterName"/> is not valid, or if size in bytes specified by <see cref="parameterValueSize"/> is less than the size of return type as described in the table above and
        /// <see cref="parameterValue"/> is not <c>null</c>.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(2, 0)]
        [DllImport("OpenCL", EntryPoint = "clGetPipeInfo")]
        public static extern Result GetPipeInformation(
            [In] IntPtr pipe,
            [In] [MarshalAs(UnmanagedType.U4)] PipeInformation parameterName,
            [In] UIntPtr parameterValueSize,
            [Out] byte[] parameterValue,
            [Out] out UIntPtr parameterValueSizeReturned
        );

        /// <summary>
        /// Registers a user callback function with a memory object.
        /// </summary>
        /// <param name="memoryObject">A valid memory object.</param>
        /// <param name="notificationCallback">
        /// The callback function that can be registered by the application. This callback function may be called asynchronously by the OpenCL implementation. It is the application's responsibility to ensure that the callback function is thread-safe.
        /// The parameters to this callback function are:
        /// memoryObject: The memory object being deleted. When the user callback is called by the implementation, this memory object is no longer valid. The memory object is only provided for reference purposes.
        /// userData: A pointer to user supplied data.
        /// </param>
        /// <param name="userData">Will be passed as the userData argument when <see param="notificationCallback"/> is called. <see cref="userData"/> can be <c>null</c>.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following:
        /// 
        /// <c>Result.InvalidMemoryObject</c> if <see cref="memoryObject"/> is not a valid memory object.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="notificationCallback"/> is not <c>null</c>.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 1)]
        [DllImport("OpenCL", EntryPoint = "clSetMemObjectDestructorCallback")]
        public static extern Result SetMemoryObjectDestructorCallback(
            [In] IntPtr memoryObject,
            [In] IntPtr notificationCallback,
            [In] IntPtr userData
        );

        #endregion

        #region Deprecated Public Methods

        /// <summary>
        /// Creates a 2D image object.
        /// </summary>
        /// <param name="context">A valid OpenCL context on which the image object is to be created.</param>
        /// <param name="flags">An enumeration that is used to specify allocation and usage information about the image memory object being created.</param>
        /// <param name="imageFormat">A pointer to a structure that describes format properties of the image to be allocated.</param>
        /// <param name="imageWidth">The widthof the image in pixels. This must be values greater than or equal to 1.</param>
        /// <param name="imageHeight">The height of the image in pixels. This must be values greater than or equal to 1.</param>
        /// <param name="imageRowPitch">
        /// The scan-line pitch in bytes. This must be 0 if <see cref="hostPointer"/> is <c>null</c> and can be either 0 or greater than or equal to <see cref="imageWidth"/> * size of element in bytes if <see cref="hostPointer"/> is not <c>null</c>.
        /// If <see cref="hostPointer"/> is not <c>null</c> and <see cref="imageRowPitch"/> is equal to 0, <see cref="imageRowPitch"/> is calculated as <see cref="imageWidth"/> * size of element in bytes. If  <see cref="imageRowPitch"/> is not 0,
        /// it must be a multiple of the image element size in bytes.
        /// </param>
        /// <param name="hostPointer">
        /// A pointer to the image data that may already be allocated by the application. The size of the buffer that <see cref="hostPointer"/> points to must be greater than or equal to <see cref="imageRowPitch"/> * <see cref="imageHeight"/>.
        /// The size of each element in bytes must be a power of 2. The image data specified by <see cref="hostPointer"/> is stored as a linear sequence of adjacent scanlines. Each scanline is stored as a linear sequence of image elements.
        /// </param>
        /// <param name="errorCode">Will return an appropriate error code. If <see cref="errorCode"/> is <c>null</c>, no error code is returned.</param>
        /// <returns>
        /// Returns a valid non-zero image object and <see cref="errorCode"/> is set to <c>Result.Success</c> if the image object is created successfully. Otherwise, it returns a <c>null</c> value with an error value returned in
        /// <see cref="errorCode"/>.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clCreateImage2D")]
        [Obsolete("This is a deprecated OpenCL 1.1 method, please use CreateImage instead.")]
        public static extern IntPtr CreateImage2D(
            [In] IntPtr context,
            [In] [MarshalAs(UnmanagedType.U8)] MemoryFlag flags,
            [In] IntPtr imageFormat,
            [In] UIntPtr imageWidth,
            [In] UIntPtr imageHeight,
            [In] UIntPtr imageRowPitch,
            [In] IntPtr hostPointer,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        /// <summary>
        /// Creates a 3D image object.
        /// </summary>
        /// <param name="context">A valid OpenCL context on which the image object is to be created.</param>
        /// <param name="flags">An enumeration that is used to specify allocation and usage information about the image memory object being created.</param>
        /// <param name="imageFormat">A pointer to a structure that describes format properties of the image to be allocated.</param>
        /// <param name="imageWidth">The widthof the image in pixels. This must be values greater than or equal to 1.</param>
        /// <param name="imageHeight">The height of the image in pixels. This must be values greater than or equal to 1.</param>
        /// <param name="imageDepth">The depth of the image in pixels. This must be values greater than or equal to 1.</param>
        /// <param name="imageRowPitch">
        /// The scan-line pitch in bytes. This must be 0 if <see cref="hostPointer"/> is <c>null</c> and can be either 0 or greater than or equal to <see cref="imageWidth"/> * size of element in bytes if <see cref="hostPointer"/> is not <c>null</c>.
        /// If <see cref="hostPointer"/> is not <c>null</c> and <see cref="imageRowPitch"/> is equal to 0, <see cref="imageRowPitch"/> is calculated as <see cref="imageWidth"/> * size of element in bytes. If  <see cref="imageRowPitch"/> is not 0,
        /// it must be a multiple of the image element size in bytes.
        /// </param>
        /// <param name="imageSlicePitch">
        /// The size in bytes of each 2D slice in the 3D image. This must be 0 if <see cref="hostPointer"/> is <c>null</c> and can be either 0 or greater than or equal to <see cref="imageRowPitch"/> * <see cref="imageHeight"/> if
        /// <see cref="hostPointer"/> is not <c>null</c>. If <see cref="hostPointer"/> is not <c>null</c> and <see cref="imageSlicePitch"/> equal to 0, <see cref="imageSlicePitch"/> is calculated as <see cref="imageRowPitch"/> *
        /// <see cref="imageHeight"/>. If <see cref="imageSlicePitch"/> is not 0, it must be a multiple of the <see cref="imageRowPitch"/>.
        /// </param>
        /// <param name="hostPointer">
        /// A pointer to the image data that may already be allocated by the application. The size of the buffer that <see cref="hostPointer"/> points to must be greater than or equal to <see cref="imageSlicePitch"/> * <see cref="imageDepth"/>.
        /// The size of each element in bytes must be a power of 2. The image data specified by <see cref="hostPointer"/> is stored as a linear sequence of adjacent 2D slices. Each 2D slice is a linear sequence of adjacent scanlines. Each scanline
        /// is a linear sequence of image elements.
        /// </param>
        /// <param name="errorCode">Will return an appropriate error code. If <see cref="errorCode"/> is <c>null</c>, no error code is returned.</param>
        /// <returns>
        /// Returns a valid non-zero image object and <see cref="errorCode"/> is set to <c>Result.Success</c> if the image object is created successfully. Otherwise, it returns a <c>null</c> value with an error value returned in
        /// <see cref="errorCode"/>.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clCreateImage3D")]
        [Obsolete("This is a deprecated OpenCL 1.1 method, please use CreateImage instead.")]
        public static extern IntPtr CreateImage3D(
            [In] IntPtr context,
            [In] [MarshalAs(UnmanagedType.U8)] MemoryFlag flags,
            [In] IntPtr imageFormat,
            [In] UIntPtr imageWidth,
            [In] UIntPtr imageHeight,
            [In] UIntPtr imageDepth,
            [In] UIntPtr imageRowPitch,
            [In] UIntPtr imageSlicePitch,
            [In] IntPtr hostPointer,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        #endregion
    }
}