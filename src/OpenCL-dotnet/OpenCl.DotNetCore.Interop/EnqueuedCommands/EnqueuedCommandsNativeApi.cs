
#region Using Directives

using System;
using System.Runtime.InteropServices;

#endregion

namespace OpenCl.DotNetCore.Interop.EnqueuedCommands
{
    /// <summary>
    /// Represents a wrapper for the native methods of the OpenCL Enqueued Commands API.
    /// </summary>
    public static class EnqueuedCommandsNativeApi
    {
        #region Public Static Methods

        /// <summary>
        /// Enqueue commands to read from a buffer object to host memory.
        /// </summary>
        /// <param name="commandQueue">Is a valid host command-queue in which the read command will be queued. commandQueue and buffer must be created with the same OpenCL context.</param>
        /// <param name="buffer">Refers to a valid buffer object.</param>
        /// <param name="blockingRead">Indicates if the read operations are blocking or non-blocking.</param>
        /// <param name="offset">The offset in bytes in the buffer object to read from.</param>
        /// <param name="size">The size in bytes of data being read.</param>
        /// <param name="pointer">The pointer to buffer in host memory where data is to be read into.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular kernel-instance. Event objects are unique and can be used to identify a particular kernel execution instance later on. If event is <c>null</c>, no event will be created for
        /// this kernel execution instance and therefore it will not be possible for the application to query or queue a wait for this particular kernel execution instance.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueReadBuffer")]
        public static extern Result EnqueueReadBuffer(
            [In] IntPtr commandQueue,
            [In] IntPtr buffer,
            [In] [MarshalAs(UnmanagedType.U4)] uint blockingRead,
            [In] UIntPtr offset,
            [In] UIntPtr size,
            [In] IntPtr pointer,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueReadBuffer")]
        public static extern Result EnqueueReadBuffer(
            [In] IntPtr commandQueue,
            [In] IntPtr buffer,
            [In] [MarshalAs(UnmanagedType.U4)] uint blockingRead,
            [In] UIntPtr offset,
            [In] UIntPtr size,
            [In] IntPtr pointer,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [In] IntPtr waitEvent
        );

        /// <summary>
        /// Enqueue command to read from a 2D or 3D rectangular region from a buffer object to host memory.
        /// </summary>
        /// <param name="commandQueue">Is is a valid host command-queue in which the read command will be queued. <see cref="commandQueue"/> and <see cref="buffer"/> must be created with the same OpenCL context.</param>
        /// <param name="buffer">Refers to a valid buffer object.</param>
        /// <param name="blockingRead">
        /// Indicates if the read operations are blocking or non-blocking.
        /// If <see cref="blockingRead"/> is <c>true</c> (1) i.e. the read command is blocking, <see cref="EnqueueReadBufferRectangle"/> does not return until the buffer data has been read and copied into memory pointed to by <see cref="pointer"/>.
        /// If <see cref="blockingRead"/> is <c>false</c> (0) i.e. the read command is non-blocking, <see cref="EnqueueReadBufferRectangle"/> queues a non-blocking read command and returns. The contents of the buffer that <see cref="pointer"/>
        /// points to cannot be used until the read command has completed. The event argument argument returns an event object which can be used to query the execution status of the read command. When the read command has completed, the contents
        /// of the buffer that <see cref="pointer"/> points to can be used by the application.
        /// </param>
        /// <param name="bufferOrigin">
        /// The (x, y, z) offset in the memory region associated with <see cref="buffer"/>. For a 2D rectangle region, the z value given by <c>bufferOrigin[2]</c> should be 0. The offset in bytes is computed as
        /// <c>bufferOrigin[2] * bufferSlicePitch + bufferOrigin[1] * bufferRowPitch + bufferOrigin[0]</c>.
        /// </param>
        /// <param name="hostOrigin">
        /// The (x, y, z) offset in the memory region pointed to by <see cref="pointer"/>. For a 2D rectangle region, the z value given by <c>hostOrigin[2]</c> should be 0. The offset in bytes is computed as
        /// <c>hostOrigin[2] * hostSlicePitch + hostOrigin[1] * hostRowPitch + hostOrigin[0]</c>.
        /// </param>
        /// <param name="region">
        /// The (width in bytes, height in rows, depth in slices) of the 2D or 3D rectangle being read or written. For a 2D rectangle copy, the depth value given by <c>region[2]</c> should be 1. The values in <see cref="region"/> cannot be 0.
        /// </param>
        /// <param name="bufferRowPitch">The length of each row in bytes to be used for the memory region associated with <see cref="buffer"/>. If <see cref="bufferRowPitch"/> is 0, <see cref="bufferRowPitch"/> is computed as <c>region[0]</c>.</param>
        /// <param name="bufferSlicePitch">
        /// The length of each 2D slice in bytes to be used for the memory region associated with <see cref="buffer"/>. If <see cref="bufferSlicePitch"/> is 0, <see cref="bufferSlicePitch"/> is computed as <c>region[1] * bufferRowPitch</c>.
        /// </param>
        /// <param name="hostRowPitch">The length of each row in bytes to be used for the memory region pointed to by <see cref="pointer"/>. If <see cref="hostRowPitch"/> is 0, <see cref="hostRowPitch"/> is computed as <c>region[0]</c>.</param>
        /// <param name="hostSlicePitch">
        /// The length of each 2D slice in bytes to be used for the memory region pointed to by <see cref="pointer"/>. If <see cref="hostSlicePitch"/> is 0, <see cref="hostSlicePitch"/> is computed as <c>region[1] * hostRowPitch</c>.
        /// </param>
        /// <param name="pointer">The pointer to buffer in host memory where data is to be read into.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular read command and can be used to query or queue a wait for this particular command to complete. event can be <c>null</c> in which case it will not be possible for the application
        /// to query the status of this command or queue a wait for this command to complete. If the <see cref="eventWaitList"/> and the event arguments are not <c>null</c>, the event argument should not refer to an element of the
        /// <see cref="eventWaitList"/> array.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 1)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueReadBufferRect")]
        public static extern Result EnqueueReadBufferRectangle(
            [In] IntPtr commandQueue,
            [In] IntPtr buffer,
            [In] [MarshalAs(UnmanagedType.U4)] uint blockingRead,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] bufferOrigin,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] hostOrigin,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] region,
            [In] UIntPtr bufferRowPitch,
            [In] UIntPtr bufferSlicePitch,
            [In] UIntPtr hostRowPitch,
            [In] UIntPtr hostSlicePitch,
            [In] IntPtr pointer,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueue commands to write to a buffer object from host memory.
        /// </summary>
        /// <param name="commandQueue">Is a valid host command-queue in which the write command will be queued. <see cref="commandQueue"/> and <see cref="buffer"/> must be created with the same OpenCL context.</param>
        /// <param name="buffer">Refers to a valid buffer object.</param>
        /// <param name="blockingWrite">
        /// Indicates if the write operations are blocking or non-blocking.
        /// If <see cref="blockingWrite"/> is <c>true</c> (1), the OpenCL implementation copies the data referred to by <see cref="pointer"/> and enqueues the write operation in the command-queue. The memory pointed to by <see cref="pointer"/> can
        /// be reused by the application after the <see cref="EnqueueWriteBuffer"/> call returns.
        /// If blocking_write is <c>false</c> (0), the OpenCL implementation will use <see cref="pointer"/> to perform a non-blocking write. As the write is non-blocking the implementation can return immediately. The memory pointed to by
        /// <see cref="pointer"/> cannot be reused by the application after the call returns. The <see cref="event"/> argument returns an event object which can be used to query the execution status of the write command. When the write command
        /// has completed, the memory pointed to by <see cref="pointer"/> can then be reused by the application.
        /// </param>
        /// <param name="offset">The offset in bytes in the buffer object to write to.</param>
        /// <param name="size">The size in bytes of data being written.</param>
        /// <param name="pointer">The pointer to buffer in host memory where data is to be written from.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular write command and can be used to query or queue a wait for this particular command to complete. event can be <c>null</c> in which case it will not be possible for the application
        /// to query the status of this command or queue a wait for this command to complete. If the <see cref="eventWaitList"/> and the event arguments are not <c>null</c>, the event argument should not refer to an element of the
        /// <see cref="eventWaitList"/> array.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueWriteBuffer")]
        public static extern Result EnqueueWriteBuffer(
            [In] IntPtr commandQueue,
            [In] IntPtr buffer,
            [In] [MarshalAs(UnmanagedType.U4)] uint blockingWrite,
            [In] UIntPtr offset,
            [In] UIntPtr size,
            [In] IntPtr pointer,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueWriteBuffer")]
        public static extern Result EnqueueWriteBuffer(
            [In] IntPtr commandQueue,
            [In] IntPtr buffer,
            [In] [MarshalAs(UnmanagedType.U4)] uint blockingWrite,
            [In] UIntPtr offset,
            [In] UIntPtr size,
            [In] IntPtr pointer,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [In] IntPtr waitEvent
        );

        /// <summary>
        /// Enqueue command to write a 2D or 3D rectangular region to a buffer object from host memory.
        /// </summary>
        /// <param name="commandQueue">Is a valid host command-queue in which the write command will be queued. <see cref="commandQueue"/> and <see cref="buffer"/> must be created with the same OpenCL context.</param>
        /// <param name="buffer">Refers to a valid buffer object.</param>
        /// <param name="blockingWrite">
        /// Indicates if the write operations are blocking or non-blocking.
        /// If <see cref="blockingWrite"/> is <c>true</c> (1), the OpenCL implementation copies the data referred to by <see cref="pointer"/> and enqueues the write operation in the command-queue. The memory pointed to by <see cref="pointer"/> can
        /// be reused by the application after the <see cref="clEnqueueWriteBufferRect"/> call returns.
        /// If blocking_write is <c>false</c> (0), the OpenCL implementation will use <see cref="pointer"/> to perform a non-blocking write. As the write is non-blocking the implementation can return immediately. The memory pointed to by
        /// <see cref="pointer"/> cannot be reused by the application after the call returns. The <see cref="event"/> argument returns an event object which can be used to query the execution status of the write command. When the write command
        /// has completed, the memory pointed to by <see cref="pointer"/> can then be reused by the application.
        /// </param>
        /// <param name="bufferOrigin">
        /// The (x, y, z) offset in the memory region associated with <see cref="buffer"/>. For a 2D rectangle region, the z value given by <c>bufferOrigin[2]</c> should be 0. The offset in bytes is computed as
        /// <c>bufferOrigin[2] * bufferSlicePitch + bufferOrigin[1] * bufferRowPitch + bufferOrigin[0]</c>.
        /// </param>
        /// <param name="hostOrigin">
        /// The (x, y, z) offset in the memory region pointed to by <see cref="pointer"/>. For a 2D rectangle region, the z value given by <c>hostOrigin[2]</c> should be 0. The offset in bytes is computed as
        /// <c>hostOrigin[2] * hostSlicePitch + hostOrigin[1] * hostRowPitch + hostOrigin[0]</c>.
        /// </param>
        /// <param name="region">
        /// The (width in bytes, height in rows, depth in slices) of the 2D or 3D rectangle being read or written. For a 2D rectangle copy, the depth value given by <c>region[2]</c> should be 1. The values in <see cref="region"/> cannot be 0.
        /// </param>
        /// <param name="bufferRowPitch">The length of each row in bytes to be used for the memory region associated with <see cref="buffer"/>. If <see cref="bufferRowPitch"/> is 0, <see cref="bufferRowPitch"/> is computed as <c>region[0]</c>.</param>
        /// <param name="bufferSlicePitch">
        /// The length of each 2D slice in bytes to be used for the memory region associated with <see cref="buffer"/>. If <see cref="bufferSlicePitch"/> is 0, <see cref="bufferSlicePitch"/> is computed as <c>region[1] * bufferRowPitch</c>.
        /// </param>
        /// <param name="hostRowPitch">The length of each row in bytes to be used for the memory region pointed to by <see cref="pointer"/>. If <see cref="hostRowPitch"/> is 0, <see cref="hostRowPitch"/> is computed as <c>region[0]</c>.</param>
        /// <param name="hostSlicePitch">
        /// The length of each 2D slice in bytes to be used for the memory region pointed to by <see cref="pointer"/>. If <see cref="hostSlicePitch"/> is 0, <see cref="hostSlicePitch"/> is computed as <c>region[1] * hostRowPitch</c>.
        /// </param>
        /// <param name="pointer">The pointer to buffer in host memory where data is to be written from.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular write command and can be used to query or queue a wait for this particular command to complete. event can be <c>null</c> in which case it will not be possible for the application
        /// to query the status of this command or queue a wait for this command to complete. If the <see cref="eventWaitList"/> and the event arguments are not <c>null</c>, the event argument should not refer to an element of the
        /// <see cref="eventWaitList"/> array.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 1)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueWriteBufferRect")]
        public static extern Result EnqueueWriteBufferRectangle(
            [In] IntPtr commandQueue,
            [In] IntPtr buffer,
            [In] [MarshalAs(UnmanagedType.U4)] uint blockingWrite,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] bufferOrigin,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] hostOrigin,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] region,
            [In] UIntPtr bufferRowPitch,
            [In] UIntPtr bufferSlicePitch,
            [In] UIntPtr hostRowPitch,
            [In] UIntPtr hostSlicePitch,
            [In] IntPtr pointer,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a command to fill a buffer object with a pattern of a given pattern size.
        /// </summary>
        /// <param name="commandQueue">Is a valid host command-queue in which the write command will be queued. <see cref="commandQueue"/> and <see cref="buffer"/> must be created with the same OpenCL context.</param>
        /// <param name="buffer">Refers to a valid buffer object.</param>
        /// <param name="pattern">
        /// A pointer to the data pattern of size <see cref="patternSize"/> in bytes. pattern will be used to fill a region in <see cref="buffer"/> starting at <see cref="offset"/> and is <see cref="size"/> bytes in size. The data pattern must be
        /// a scalar or vector integer or floating-point data type. For example, if <see cref="buffer"/> is to be filled with a pattern of <c>float4</c> values, then <see cref="pattern"/> will be a pointer to a <c>float4</c> value and
        /// <see cref="patternSize"/> will be <c>sizeof(float4)</c>. The maximum value of <see cref="patternSize"/> is the size of the largest integer or floating-point vector data type supported by the OpenCL device. The memory associated with
        /// <see cref="pattern"/> can be reused or freed after the function returns.</param>
        /// <param name="patternSize">The size of <see cref="pattern"/> in bytes.</param>
        /// <param name="offset">The location in bytes of the region being filled in <see cref="buffer"/> and must be a multiple of <see cref="patternSize"/>.</param>
        /// <param name="size">The size in bytes of region being filled in <see cref="buffer"/> and must be a multiple of <see cref="patternSize"/>.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular write command and can be used to query or queue a wait for this particular command to complete. event can be <c>null</c> in which case it will not be possible for the application
        /// to query the status of this command or queue a wait for this command to complete. If the <see cref="eventWaitList"/> and the event arguments are not <c>null</c>, the event argument should not refer to an element of the
        /// <see cref="eventWaitList"/> array.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 2)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueFillBuffer")]
        public static extern Result EnqueueFillBuffer(
            [In] IntPtr commandQueue,
            [In] IntPtr buffer,
            [In] IntPtr pattern,
            [In] UIntPtr patternSize,
            [In] UIntPtr offset,
            [In] UIntPtr size,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        [IntroducedInOpenCl(1, 2)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueFillBuffer")]
        public static extern Result EnqueueFillBuffer(
            [In] IntPtr commandQueue,
            [In] IntPtr buffer,
            [In] IntPtr pattern,
            [In] UIntPtr patternSize,
            [In] UIntPtr offset,
            [In] UIntPtr size,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [In] IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a command to copy from one buffer object to another.
        /// </summary>
        /// <param name="commandQueue">Is a valid host command-queue in which the write command will be queued. <see cref="commandQueue"/> and <see cref="buffer"/> must be created with the same OpenCL context.</param>
        /// <param name="sourceBuffer">A valid source buffer object.</param>
        /// <param name="destinationBuffer">A valid destination buffer object.</param>
        /// <param name="sourceOffset">The offset where to begin copying data from <see cref="sourceBuffer"/>.</param>
        /// <param name="destinationOffset">The offset where to begin copying data into <see cref="destinationBuffer"/>.</param>
        /// <param name="size">Refers to the size in bytes to copy.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular write command and can be used to query or queue a wait for this particular command to complete. event can be <c>null</c> in which case it will not be possible for the application
        /// to query the status of this command or queue a wait for this command to complete. If the <see cref="eventWaitList"/> and the event arguments are not <c>null</c>, the event argument should not refer to an element of the
        /// <see cref="eventWaitList"/> array.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueCopyBuffer")]
        public static extern Result EnqueueCopyBuffer(
            [In] IntPtr commandQueue,
            [In] IntPtr sourceBuffer,
            [In] IntPtr destinationBuffer,
            [In] UIntPtr sourceOffset,
            [In] UIntPtr destinationOffset,
            [In] UIntPtr size,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a command to copy a 2D or 3D rectangular region from a buffer object to another buffer object.
        /// </summary>
        /// <param name="commandQueue">
        /// The host command-queue in which the copy command will be queued. The OpenCL context associated with <see cref="commandQueue"/>, <see cref="sourceBuffer"/>, and <see cref="destinationBuffer"/> must be the same.
        /// </param>
        /// <param name="sourceBuffer">A valid source buffer object.</param>
        /// <param name="destinationBuffer">A valid destination buffer object.</param>
        /// <param name="sourceOrigin">
        /// The (x, y, z) offset in the memory region associated with <see cref="sourceBuffer"/>. For a 2D rectangle region, the z value given by <c>sourceOrigin[2]</c> should be 0. The offset in bytes is computed as
        /// <c>sourceOrigin[2] * sourceSlicePitch + sourceOrigin[1] * sourceRowPitch + sourceOrigin[0]</c>.
        /// </param>
        /// <param name="destinationOrigin">
        /// The (x, y, z) offset in the memory region associated with <see cref="destinationBuffer"/>. For a 2D rectangle region, the z value given by <c>destinationOrigin[2]</c> should be 0. The offset in bytes is computed as
        /// <c>destinationOrigin[2] * destinationSlicePitch + destinationOrigin[1] * destinationRowPitch + destinationOrigin[0]</c>.
        /// </param>
        /// <param name="region">
        /// The (width in bytes, height in rows, depth in slices) in bytes of the 2D or 3D rectangle being copied. For a 2D rectangle, the depth value given by <c>region[2]</c> should be 1. The values in <see cref="region"/> cannot be 0.
        /// </param>
        /// <param name="sourceRowPitch">
        /// The length of each row in bytes to be used for the memory region associated with <see cref="sourceBuffer"/>. If <see cref="sourceRowPitch"/> is 0, <see cref="sourceRowPitch"/> is computed as <c>region[0]</c>.
        /// </param>
        /// <param name="sourceSlicePitch">
        /// The length of each 2D slice in bytes to be used for the memory region associated with <see cref="sourceBuffer"/>. If <see cref="sourceSlicePitch"/> is 0, <see cref="sourceSlicePitch"/> is computed as <c>region[1] * sourceRowPitch</c>.
        /// </param>
        /// <param name="destinationRowPitch">
        /// The length of each row in bytes to be used for the memory region associated with <see cref="destinationBuffer"/>. If <see cref="destinationRowPitch"/> is 0, <see cref="destinationRowPitch"/> is computed as <c>region[0]</c>.
        /// </param>
        /// <param name="destinationSlicePitch">
        /// The length of each 2D slice in bytes to be used for the memory region associated with <see cref="destinationBuffer"/>. If <see cref="destinationSlicePitch"/> is 0, <see cref="destinationSlicePitch"/> is computed as
        /// <c>region[1] * destinationRowPitch</c>.
        /// </param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular write command and can be used to query or queue a wait for this particular command to complete. event can be <c>null</c> in which case it will not be possible for the application
        /// to query the status of this command or queue a wait for this command to complete. If the <see cref="eventWaitList"/> and the event arguments are not <c>null</c>, the event argument should not refer to an element of the
        /// <see cref="eventWaitList"/> array.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 1)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueCopyBufferRect")]
        public static extern Result EnqueueCopyBufferRectangle(
            [In] IntPtr commandQueue,
            [In] IntPtr sourceBuffer,
            [In] IntPtr destinationBuffer,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] sourceOrigin,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] destinationOrigin,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] region,
            [In] UIntPtr sourceRowPitch,
            [In] UIntPtr sourceSlicePitch,
            [In] UIntPtr destinationRowPitch,
            [In] UIntPtr destinationSlicePitch,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueue commands to read from an image or image array object to host memory.
        /// </summary>
        /// <param name="commandQueue">Is a valid host command-queue in which the read command will be queued. commandQueue and buffer must be created with the same OpenCL context.</param>
        /// <param name="image">Refers to a valid image or image array object.</param>
        /// <param name="blockingRead">
        /// Indicates if the read operations are blocking or non-blocking.
        /// If <see cref="blockingRead"/> is <c>true</c> (1) i.e. the read command is blocking, <see cref="EnqueueReadImage"/> does not return until the buffer data has been read and copied into memory pointed to by <see cref="pointer"/>.
        /// If <see cref="blockingRead"/> is <c>false</c> (0) i.e. the read command is non-blocking, <see cref="EnqueueReadImage"/> queues a non-blocking read command and returns. The contents of the buffer that <see cref="pointer"/>
        /// points to cannot be used until the read command has completed. The event argument argument returns an event object which can be used to query the execution status of the read command. When the read command has completed, the contents
        /// of the buffer that <see cref="pointer"/> points to can be used by the application.
        /// </param>
        /// <param name="origin">
        /// Defines the (x, y, z) offset in pixels in the 1D, 2D, or 3D image, the (x, y) offset and the image index in the image array or the (x) offset and the image index in the 1D image array. If <see cref="image"/> is a 2D image object,
        /// <c>origin[2]</c> must be 0. If <see cref="image"/> is a 1D image or 1D image buffer object, <c>origin[1]</c> and <c>origin[2]</c> must be 0. If <see cref="image"/> is a 1D image array object, <c>origin[2]</c> must be 0. If
        /// <see cref="image"/> is a 1D image array object, <c>origin[1]</c> describes the image index in the 1D image array. If <see cref="image"/> is a 2D image array object, <c>origin[2]</c> describes the image index in the 2D image array.
        /// </param>
        /// <param name="region">
        /// Defines the (width, height, depth) in pixels of the 1D, 2D or 3D rectangle, the (width, height) in pixels of the 2D rectangle and the number of images of a 2D image array or the (width) in pixels of the 1D rectangle and the number of
        /// images of a 1D image array. If image is a 2D image object, region[2] must be 1. If image is a 1D image or 1D image buffer object, region[1] and region[2] must be 1. If image is a 1D image array object, region[2] must be 1. The values in region cannot be 0.
        /// </param>
        /// <param name="rowPitch">
        /// The length of each row in bytes. This value must be greater than or equal to the element size in bytes * width. If <see cref="rowPitch"/> is set to 0, the appropriate row pitch is calculated based on the size of each element in bytes
        /// multiplied by width.
        /// </param>
        /// <param name="slicePitch">
        /// Size in bytes of the 2D slice of the 3D region of a 3D image or each image of a 1D or 2D image array being read. This must be 0 if <see cref="image"/> is a 1D or 2D image. Otherwise this value must be greater than or equal to
        /// <c>rowPitch * height</c>. If <see cref="slicePitch"/> is set to 0, the appropriate slice pitch is calculated based on the <c>rowPitch * height</c>.
        /// </param>
        /// <param name="pointer">The pointer to a buffer in host memory where image data is to be read from.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular write command and can be used to query or queue a wait for this particular command to complete. event can be <c>null</c> in which case it will not be possible for the application
        /// to query the status of this command or queue a wait for this command to complete. If the <see cref="eventWaitList"/> and the event arguments are not <c>null</c>, the event argument should not refer to an element of the
        /// <see cref="eventWaitList"/> array.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueReadImage")]
        public static extern Result EnqueueReadImage(
            [In] IntPtr commandQueue,
            [In] IntPtr image,
            [In] [MarshalAs(UnmanagedType.U4)] uint blockingRead,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] origin,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] region,
            [In] UIntPtr rowPitch,
            [In] UIntPtr slicePitch,
            [In] IntPtr pointer,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a command to write to an image or image array object from host memory.
        /// </summary>
        /// <param name="commandQueue">Refers to the host command-queue in which the write command will be queued. <see cref="commandQueue"/> and <see cref="image"/> must be created with the same OpenCL context.</param>
        /// <param name="image">Refers to a valid image or image array object.</param>
        /// <param name="blockingWrite">
        /// Indicates if the write operations are blocking or non-blocking.
        /// If <see cref="blockingWrite"/> is <c>true</c> (1), the OpenCL implementation copies the data referred to by <see cref="pointer"/> and enqueues the write operation in the command-queue. The memory pointed to by <see cref="pointer"/> can
        /// be reused by the application after the <see cref="EnqueueWriteImage"/> call returns.
        /// If blocking_write is <c>false</c> (0), the OpenCL implementation will use <see cref="pointer"/> to perform a non-blocking write. As the write is non-blocking the implementation can return immediately. The memory pointed to by
        /// <see cref="pointer"/> cannot be reused by the application after the call returns. The <see cref="event"/> argument returns an event object which can be used to query the execution status of the write command. When the write command
        /// has completed, the memory pointed to by <see cref="pointer"/> can then be reused by the application.
        /// </param>
        /// <param name="origin">
        /// Defines the (x, y, z) offset in pixels in the 1D, 2D, or 3D image, the (x, y) offset and the image index in the image array or the (x) offset and the image index in the 1D image array. If <see cref="image"/> is a 2D image object,
        /// <c>origin[2]</c> must be 0. If <see cref="image"/> is a 1D image or 1D image buffer object, <c>origin[1]</c> and <c>origin[2]</c> must be 0. If <see cref="image"/> is a 1D image array object, <c>origin[2]</c> must be 0. If
        /// <see cref="image"/> is a 1D image array object, <c>origin[1]</c> describes the image index in the 1D image array. If <see cref="image"/> is a 2D image array object, <c>origin[2]</c> describes the image index in the 2D image array.
        /// </param>
        /// <param name="region">
        /// Defines the (width, height, depth) in pixels of the 1D, 2D or 3D rectangle, the (width, height) in pixels of the 2D rectangle and the number of images of a 2D image array or the (width) in pixels of the 1D rectangle and the number of
        /// images of a 1D image array. If image is a 2D image object, region[2] must be 1. If image is a 1D image or 1D image buffer object, region[1] and region[2] must be 1. If image is a 1D image array object, region[2] must be 1. The values in region cannot be 0.
        /// </param>
        /// <param name="inputRowPitch">
        /// The length of each row in bytes. This value must be greater than or equal to the element size in bytes * width. If <see cref="inputRowPitch"/> is set to 0, the appropriate row pitch is calculated based on the size of each element in bytes
        /// multiplied by width.
        /// </param>
        /// <param name="inputSlicePitch">
        /// Size in bytes of the 2D slice of the 3D region of a 3D image or each image of a 1D or 2D image array being read. This must be 0 if <see cref="image"/> is a 1D or 2D image. Otherwise this value must be greater than or equal to
        /// <c>inputRowPitch * height</c>. If <see cref="inputSlicePitch"/> is set to 0, the appropriate slice pitch is calculated based on the <c>inputRowPitch * height</c>.
        /// </param>
        /// <param name="pointer">The pointer to a buffer in host memory where image data is to be written to.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular write command and can be used to query or queue a wait for this particular command to complete. event can be <c>null</c> in which case it will not be possible for the application
        /// to query the status of this command or queue a wait for this command to complete. If the <see cref="eventWaitList"/> and the event arguments are not <c>null</c>, the event argument should not refer to an element of the
        /// <see cref="eventWaitList"/> array.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueWriteImage")]
        public static extern Result EnqueueWriteImage(
            [In] IntPtr commandQueue,
            [In] IntPtr image,
            [In] [MarshalAs(UnmanagedType.U4)] uint blockingWrite,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] origin,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] region,
            [In] UIntPtr inputRowPitch,
            [In] UIntPtr inputSlicePitch,
            [In] IntPtr pointer,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a command to fill an image object with a specified color.
        /// </summary>
        /// <param name="commandQueue">Refers to the host command-queue in which the fill command will be queued. The OpenCL context associated with <see cref="commandQueue"/> and <see cref="image"/> must be the same.</param>
        /// <param name="image">A valid image object.</param>
        /// <param name="fillColor">
        /// The color used to fill the image. The fill color is a single floating point value if the channel order is <c>ChannelOrder.Depth</c>. Otherwise, the fill color is a four component RGBA floating-point color value if the image channel
        /// data type is not an unnormalized signed or unsigned integer type, is a four component signed integer value if the image channel data type is an unnormalized signed integer type and is a four component unsigned integer value if the image
        /// channel data type is an unnormalized unsigned integer type. The fill color will be converted to the appropriate image channel format and order associated with image as described in sections 6.12.14 and 8.3.
        /// </param>
        /// <param name="origin">
        /// Defines the (x, y, z) offset in pixels in the 1D, 2D, or 3D image, the (x, y) offset and the image index in the image array or the (x) offset and the image index in the 1D image array. If <see cref="image"/> is a 2D image object,
        /// <c>origin[2]</c> must be 0. If <see cref="image"/> is a 1D image or 1D image buffer object, <c>origin[1]</c> and <c>origin[2]</c> must be 0. If <see cref="image"/> is a 1D image array object, <c>origin[2]</c> must be 0. If
        /// <see cref="image"/> is a 1D image array object, <c>origin[1]</c> describes the image index in the 1D image array. If <see cref="image"/> is a 2D image array object, <c>origin[2]</c> describes the image index in the 2D image array.
        /// </param>
        /// <param name="region">
        /// Defines the (width, height, depth) in pixels of the 1D, 2D or 3D rectangle, the (width, height) in pixels of the 2D rectangle and the number of images of a 2D image array or the (width) in pixels of the 1D rectangle and the number of
        /// images of a 1D image array. If image is a 2D image object, region[2] must be 1. If image is a 1D image or 1D image buffer object, region[1] and region[2] must be 1. If image is a 1D image array object, region[2] must be 1. The values in region cannot be 0.
        /// </param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular write command and can be used to query or queue a wait for this particular command to complete. event can be <c>null</c> in which case it will not be possible for the application
        /// to query the status of this command or queue a wait for this command to complete. If the <see cref="eventWaitList"/> and the event arguments are not <c>null</c>, the event argument should not refer to an element of the
        /// <see cref="eventWaitList"/> array.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 2)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueFillImage")]
        public static extern Result EnqueueFillImage(
            [In] IntPtr commandQueue,
            [In] IntPtr image,
            [In] IntPtr fillColor,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] origin,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] region,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a command to copy image objects.
        /// </summary>
        /// <param name="commandQueue">
        /// Refers to the host command-queue in which the copy command will be queued. The OpenCL context associated with <see cref="commandQueue"/>, <see cref="sourceImage"/> and <see cref="destinationImage"/> must be the same.
        /// </param>
        /// <param name="sourceImage">
        /// Can be 1D, 2D, 3D image or a 1D, 2D image array objects. It is possible to copy subregions between any combinations of source and destination types, provided that the dimensions of the subregions are the same e.g., one can copy a
        /// rectangular region from a 2D image to a slice of a 3D image.
        /// </param>
        /// <param name="destinationImage">
        /// Can be 1D, 2D, 3D image or a 1D, 2D image array objects. It is possible to copy subregions between any combinations of source and destination types, provided that the dimensions of the subregions are the same e.g., one can copy a
        /// rectangular region from a 2D image to a slice of a 3D image.
        /// </param>
        /// <param name="sourceOrigin">
        /// Defines the (x, y, z) offset in pixels in the 1D, 2D or 3D image, the (x, y) offset and the image index in the 2D image array or the (x) offset and the image index in the 1D image array. If <see cref="image"/> is a 2D image object,
        /// <c>sourceOrigin[2]</c> must be 0. If <see cref="sourceImage"/> is a 1D image object, <c>sourceOrigin[1]</c> and <c>sourceOrigin[2]</c> must be 0. If <see cref="sourceImage"/> is a 1D image array object, <c>sourceOrigin[2]</c> must be 0.
        /// If <see cref="sourceImage"/> is a 1D image array object, <c>sourceOrigin[1]</c> describes the image index in the 1D image array. If <see cref="sourceImage"/> is a 2D image array object, <c>sourceOrigin[2]</c> describes the image index
        /// in the 2D image array.
        /// </param>
        /// <param name="destinationOrigin">
        /// Defines the (x, y, z) offset in pixels in the 1D, 2D or 3D image, the (x, y) offset and the image index in the 2D image array or the (x) offset and the image index in the 1D image array. If <see cref="destinationImage"/> is a 2D image
        /// object, <c>destinationOrigin[2]</c> must be 0. If <see cref="destinationImage"/> is a 1D image or 1D image buffer object, <c>destinationOrigin[1]</c> and <c>destinationOrigin[2]</c> must be 0. If <see cref="destinationImage"/> is a 1D
        /// image array object, <c>destinationOrigin[2]</c> must be 0. If <see cref="destinationImage"/> is a 1D image array object, <c>destinationOrigin[1]</c> describes the image index in the 1D image array. If <see cref="destinationImage"/> is
        /// a 2D image array object, <c>destinationOrigin[2]</c> describes the image index in the 2D image array.
        /// </param>
        /// <param name="region">
        /// Defines the (width, height, depth) in pixels of the 1D, 2D or 3D rectangle, the (width, height) in pixels of the 2D rectangle and the number of images of a 2D image array or the (width) in pixels of the 1D rectangle and the number of
        /// images of a 1D image array. If <see cref="sourceImage"/> or <see cref="destinationImage"/> is a 2D image object, <c>region[2]</c> must be 1. If <see cref="sourceImage"/> or <see cref="destinationImage"/> is a 1D image or 1D image buffer
        /// object, <c>region[1]</c> and <c>region[2]</c> must be 1. If <see cref="sourceImage"/> or <see cref="destinationImage"/> is a 1D image array object, <c>region[2]</c> must be 1. The values in region cannot be 0.
        /// </param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular write command and can be used to query or queue a wait for this particular command to complete. event can be <c>null</c> in which case it will not be possible for the application
        /// to query the status of this command or queue a wait for this command to complete. If the <see cref="eventWaitList"/> and the event arguments are not <c>null</c>, the event argument should not refer to an element of the
        /// <see cref="eventWaitList"/> array.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueCopyImage")]
        public static extern Result EnqueueCopyImage(
            [In] IntPtr commandQueue,
            [In] IntPtr sourceImage,
            [In] IntPtr destinationImage,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] sourceOrigin,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] destinationOrigin,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] region,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a command to copy an image object to a buffer object.
        /// </summary>
        /// <param name="commandQueue">Must be a valid host command-queue. The OpenCL context associated with <see cref="commandQueue"/>, <see cref="sourceImage"/>, and <see cref="destinationBuffer"/> must be the same.</param>
        /// <param name="sourceImage">A valid image object.</param>
        /// <param name="destinationBuffer">A valid buffer object.</param>
        /// <param name="sourceOrigin">
        /// Defines the (x, y, z) offset in pixels in the 1D, 2D or 3D image, the (x, y) offset and the image index in the 2D image array or the (x) offset and the image index in the 1D image array. If <see cref="image"/> is a 2D image object,
        /// <c>sourceOrigin[2]</c> must be 0. If <see cref="sourceImage"/> is a 1D image object, <c>sourceOrigin[1]</c> and <c>sourceOrigin[2]</c> must be 0. If <see cref="sourceImage"/> is a 1D image array object, <c>sourceOrigin[2]</c> must be 0.
        /// If <see cref="sourceImage"/> is a 1D image array object, <c>sourceOrigin[1]</c> describes the image index in the 1D image array. If <see cref="sourceImage"/> is a 2D image array object, <c>sourceOrigin[2]</c> describes the image index
        /// in the 2D image array.
        /// </param>
        /// <param name="region">
        /// Defines the (width, height, depth) in pixels of the 1D, 2D or 3D rectangle, the (width, height) in pixels of the 2D rectangle and the number of images of a 2D image array or the (width) in pixels of the 1D rectangle and the number of
        /// images of a 1D image array. If <see cref="sourceImage"/> or <see cref="destinationImage"/> is a 2D image object, <c>region[2]</c> must be 1. If <see cref="sourceImage"/> or <see cref="destinationImage"/> is a 1D image or 1D image buffer
        /// object, <c>region[1]</c> and <c>region[2]</c> must be 1. If <see cref="sourceImage"/> or <see cref="destinationImage"/> is a 1D image array object, <c>region[2]</c> must be 1. The values in region cannot be 0.
        /// </param>
        /// <param name="destinationOffset">
        /// Refers to the offset where to begin copying data into <see cref="destinationBuffer"/>. The size in bytes of the region to be copied referred to as dst_cb is computed as width * height * depth * bytes/image element if
        /// <see cref="sourceImage"/> is a 3D image object, is computed as width * height * bytes/image element if <see cref="sourceImage"/> is a 2D image, is computed as width * height * arraysize * bytes/image element if <see cref="sourceImage"/>
        /// is a 2D image array object, is computed as width * bytes/image element if <see cref="sourceImage"/> is a 1D image or 1D image buffer object and is computed as width * arraysize * bytes/image element if <see cref="sourceImage"/> is a 1D
        /// image array object.
        /// </param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular write command and can be used to query or queue a wait for this particular command to complete. event can be <c>null</c> in which case it will not be possible for the application
        /// to query the status of this command or queue a wait for this command to complete. If the <see cref="eventWaitList"/> and the event arguments are not <c>null</c>, the event argument should not refer to an element of the
        /// <see cref="eventWaitList"/> array.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueCopyImageToBuffer")]
        public static extern Result EnqueueCopyImageToBuffer(
            [In] IntPtr commandQueue,
            [In] IntPtr sourceImage,
            [In] IntPtr destinationBuffer,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] sourceOrigin,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] region,
            [In] UIntPtr destinationOffset,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a command to copy a buffer object to an image object.
        /// </summary>
        /// <param name="commandQueue">Must be a valid host command-queue. The OpenCL context associated with <see cref="commandQueue"/>, <see cref="sourceBuffer"/>, and <see cref="destinationImage"/> must be the same.</param>
        /// <param name="sourceBuffer">A valid buffer object.</param>
        /// <param name="destinationImage">A valid image object.</param>
        /// <param name="sourceOffset">The offset where to begin copying data from <see cref="sourceBuffer"/>.</param>
        /// <param name="destinationOrigin">
        /// Defines the (x, y, z) offset in pixels in the 1D, 2D or 3D image, the (x, y) offset and the image index in the 2D image array or the (x) offset and the image index in the 1D image array. If <see cref="destinationImage"/> is a 2D image
        /// object, <c>destinationOrigin[2]</c> must be 0. If <see cref="destinationImage"/> is a 1D image or 1D image buffer object, <c>destinationOrigin[1]</c> and <c>destinationOrigin[2]</c> must be 0. If <see cref="destinationImage"/> is a 1D
        /// image array object, <c>destinationOrigin[2]</c> must be 0. If <see cref="destinationImage"/> is a 1D image array object, <c>destinationOrigin[1]</c> describes the image index in the 1D image array. If <see cref="destinationImage"/> is
        /// a 2D image array object, <c>destinationOrigin[2]</c> describes the image index in the 2D image array.
        /// </param>
        /// <param name="region">
        /// Defines the (width, height, depth) in pixels of the 1D, 2D or 3D rectangle, the (width, height) in pixels of the 2D rectangle and the number of images of a 2D image array or the (width) in pixels of the 1D rectangle and the number of
        /// images of a 1D image array. If <see cref="destinationImage"/> is a 2D image object, <c>region[2]</c> must be 1. If <see cref="destinationImage"/> is a 1D image or 1D image buffer object, <c>region[1]</c> and <c>region[2]</c> must be 1.
        /// If <see cref="destinationImage"/> is a 1D image array object, <c>region[2]</c> must be 1. The values in region cannot be 0.
        /// The size in bytes of the region to be copied from <see cref="sourceBuffer"/> referred to as src_cb is computed as width * height * depth * bytes/image_element if <see cref="destinationImage"/> is a 3D image object, is computed as
        /// width * height * bytes/image_element if <see cref="destinationImage"/> is a 2D image, is computed as width * height * arraysize * bytes/image_element if <see cref="destinationImage"/> is a 2D image array object, is computed as
        /// width * bytes/image_element if <see cref="destinationImage"/> is a 1D image or 1D image buffer object and is computed as width * arraysize * bytes/image_element if <see cref="destinationImage"/> is a 1D image array object.
        /// </param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular write command and can be used to query or queue a wait for this particular command to complete. event can be <c>null</c> in which case it will not be possible for the application
        /// to query the status of this command or queue a wait for this command to complete. If the <see cref="eventWaitList"/> and the event arguments are not <c>null</c>, the event argument should not refer to an element of the
        /// <see cref="eventWaitList"/> array.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueCopyBufferToImage")]
        public static extern Result EnqueueCopyBufferToImage(
            [In] IntPtr commandQueue,
            [In] IntPtr sourceBuffer,
            [In] IntPtr destinationImage,
            [In] UIntPtr sourceOffset,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] destinationOrigin,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] region,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a command to map a region of the buffer object given by buffer into the host address space and returns a pointer to this mapped region.
        /// </summary>
        /// <param name="commandQueue">Must be a valid host command-queue.</param>
        /// <param name="buffer">A valid buffer object. The OpenCL context associated with <see cref="commandQueue"/> and <see cref="buffer"/> must be the same.</param>
        /// <param name="blockingMap">
        /// Indicates if the map operation is blocking or non-blocking.
        /// If <see cref="blockingMap"/> is <c>true</c> (1), <see cref="EnqueueMapBuffer"/> does not return until the specified region in <see cref="buffer"/> is mapped into the host address space and the application can access the contents of
        /// the mapped region using the pointer returned by <see cref="EnqueueMapBuffer"/>.
        /// If <see cref="blockingMap"/> is <c>false</c> (0), i.e. map operation is non-blocking, the pointer to the mapped region returned by <see cref="EnqueueMapBuffer"/> cannot be used until the map command has completed. The <see cref="event"/>
        /// argument returns an event object which can be used to query the execution status of the map command. When the map command is completed, the application can access the contents of the mapped region using the pointer returned by
        /// <see cref="EnqueueMapBuffer"/>.
        /// </param>
        /// <param name="mapFlag">An enumeration with which determines the behavior of the map operation.</param>
        /// <param name="offset">The offset in bytes of the region in the buffer object that is being mapped.</param>
        /// <param name="size">The the size in bytes of the region in the buffer object that is being mapped.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular write command and can be used to query or queue a wait for this particular command to complete. event can be <c>null</c> in which case it will not be possible for the application
        /// to query the status of this command or queue a wait for this command to complete. If the <see cref="eventWaitList"/> and the event arguments are not <c>null</c>, the event argument should not refer to an element of the
        /// <see cref="eventWaitList"/> array.
        /// </param>
        /// <param name="errorCode">Returns an appropriate error code. If <see cref="errorCode"> is <c>null</c>, no error code is returned.</param>
        /// <returns>Returns a pointer that maps a region starting at <see cref="offset"/> and is at least <see cref="size"/> bytes in size. The result of a memory access outside this region is undefined.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueMapBuffer")]
        public static extern IntPtr EnqueueMapBuffer(
            [In] IntPtr commandQueue,
            [In] IntPtr buffer,
            [In] [MarshalAs(UnmanagedType.U4)] uint blockingMap,
            [In] [MarshalAs(UnmanagedType.U8)] MapFlag mapFlag,
            [In] UIntPtr offset,
            [In] UIntPtr size,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        /// <summary>
        /// Enqueues a command to map a region of an image object into the host address space and returns a pointer to this mapped region.
        /// </summary>
        /// <param name="commandQueue">Must be a valid host command-queue.</param>
        /// <param name="image">A valid image object. The OpenCL context associated with <see cref="commandQueue"/> and <see cref="image"/> must be the same.</param>
        /// <param name="blockingMap">
        /// Indicates if the map operation is blocking or non-blocking.
        /// If <see cref="blockingMap"/> is <c>true</c> (1), <see cref="EnqueueMapBuffer"/> does not return until the specified region in <see cref="buffer"/> is mapped into the host address space and the application can access the contents of
        /// the mapped region using the pointer returned by <see cref="EnqueueMapBuffer"/>.
        /// If <see cref="blockingMap"/> is <c>false</c> (0), i.e. map operation is non-blocking, the pointer to the mapped region returned by <see cref="EnqueueMapBuffer"/> cannot be used until the map command has completed. The <see cref="event"/>
        /// argument returns an event object which can be used to query the execution status of the map command. When the map command is completed, the application can access the contents of the mapped region using the pointer returned by
        /// <see cref="EnqueueMapBuffer"/>.
        /// </param>
        /// <param name="mapFlag">An enumeration with which determines the behavior of the map operation.</param>
        /// <param name="origin">
        /// Defines the (x, y, z) offset in pixels in the 1D, 2D, or 3D image, the (x, y) offset and the image index in the image array or the (x) offset and the image index in the 1D image array. If <see cref="image"/> is a 2D image object,
        /// <c>origin[2]</c> must be 0. If <see cref="image"/> is a 1D image or 1D image buffer object, <c>origin[1]</c> and <c>origin[2]</c> must be 0. If <see cref="image"/> is a 1D image array object, <c>origin[2]</c> must be 0. If
        /// <see cref="image"/> is a 1D image array object, <c>origin[1]</c> describes the image index in the 1D image array. If <see cref="image"/> is a 2D image array object, <c>origin[2]</c> describes the image index in the 2D image array.
        /// </param>
        /// <param name="region">
        /// Defines the (width, height, depth) in pixels of the 1D, 2D or 3D rectangle, the (width, height) in pixels of the 2D rectangle and the number of images of a 2D image array or the (width) in pixels of the 1D rectangle and the number of
        /// images of a 1D image array. If image is a 2D image object, region[2] must be 1. If image is a 1D image or 1D image buffer object, region[1] and region[2] must be 1. If image is a 1D image array object, region[2] must be 1. The values in region cannot be 0.
        /// </param>
        /// <param name="inputRowPitch">
        /// The length of each row in bytes. This value must be greater than or equal to the element size in bytes * width. If <see cref="inputRowPitch"/> is set to 0, the appropriate row pitch is calculated based on the size of each element in bytes
        /// multiplied by width.
        /// </param>
        /// <param name="inputSlicePitch">
        /// Size in bytes of the 2D slice of the 3D region of a 3D image or each image of a 1D or 2D image array being read. This must be 0 if <see cref="image"/> is a 1D or 2D image. Otherwise this value must be greater than or equal to
        /// <c>inputRowPitch * height</c>. If <see cref="inputSlicePitch"/> is set to 0, the appropriate slice pitch is calculated based on the <c>inputRowPitch * height</c>.
        /// </param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular write command and can be used to query or queue a wait for this particular command to complete. event can be <c>null</c> in which case it will not be possible for the application
        /// to query the status of this command or queue a wait for this command to complete. If the <see cref="eventWaitList"/> and the event arguments are not <c>null</c>, the event argument should not refer to an element of the
        /// <see cref="eventWaitList"/> array.
        /// </param>
        /// <param name="errorCode">Returns an appropriate error code. If <see cref="errorCode"> is <c>null</c>, no error code is returned.</param>
        /// <returns>
        /// Returns a pointer that maps a 1D, 2D or 3D region starting at <see cref="origin"/> and is at least <c>region[0]</c> pixels in size for a 1D image, 1D image buffer or 1D image array, <c>(imageRowPitch * region[1])</c> pixels in size for
        /// a 2D image or 2D image array, and <c>(imageSlicePitch * region[2])</c> pixels in size for a 3D image. The result of a memory access outside this region is undefined.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueMapImage")]
        public static extern IntPtr EnqueueMapImage(
            [In] IntPtr commandQueue,
            [In] IntPtr image,
            [In] [MarshalAs(UnmanagedType.U4)] uint blockingMap,
            [In] [MarshalAs(UnmanagedType.U8)] MapFlag mapFlag,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] origin,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] region,
            [In] UIntPtr imageRowPitch,
            [In] UIntPtr imageSlicePitch,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        /// <summary>
        /// Enqueues a command to unmap a previously mapped region of a memory object.
        /// </summary>
        /// <param name="commandQueue">Must be a valid host command-queue.</param>
        /// <param name="memoryObject">A valid memory (buffer or image) object. The OpenCL context associated with <see cref="commandQueue"/> and <see cref="memoryObject"/> must be the same.</param>
        /// <param name="mappedPointer">The host address returned by a previous call to <see cref="EnqueueMapBuffer"/> or <see cref="EnqueueMapImage"/> for <see cref="memoryObject"/>.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular write command and can be used to query or queue a wait for this particular command to complete. event can be <c>null</c> in which case it will not be possible for the application
        /// to query the status of this command or queue a wait for this command to complete. If the <see cref="eventWaitList"/> and the event arguments are not <c>null</c>, the event argument should not refer to an element of the
        /// <see cref="eventWaitList"/> array.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueUnmapMemObject")]
        public static extern Result EnqueueUnmapMemoryObject(
            [In] IntPtr commandQueue,
            [In] IntPtr memoryObject,
            [In] IntPtr mappedPointer,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a command to indicate which device a set of memory objects should be associated with.
        /// </summary>
        /// <param name="commandQueue">
        /// A valid host command-queue. The specified set of memory objects in <see creF="memoryObjects"/> will be migrated to the OpenCL device associated with <see cref="commandQueue"/> or to the host if the <c>MemoryMigrationFlag.Host</c> has
        /// been specified.
        /// </param>
        /// <param name="numberOfMemoryObjects">The number of memory objects specified in <see creF="memoryObjects"/>.</param>
        /// <param name="memoryObjects">A pointer to a list of memory objects.</param>
        /// <param name="memoryMigrationFlags">An enumration that is used to specify migration options.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular write command and can be used to query or queue a wait for this particular command to complete. event can be <c>null</c> in which case it will not be possible for the application
        /// to query the status of this command or queue a wait for this command to complete. If the <see cref="eventWaitList"/> and the event arguments are not <c>null</c>, the event argument should not refer to an element of the
        /// <see cref="eventWaitList"/> array.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 2)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueMigrateMemObjects")]
        public static extern Result EnqueueMigrateMemorysObjects(
            [In] IntPtr commandQueue,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfMemoryObjects,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] memoryObjects,
            [In] [MarshalAs(UnmanagedType.U8)] MemoryMigrationFlag memoryMigrationFlags,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a command to execute a kernel on a device.
        /// </summary>
        /// <param name="commandQueue">A valid host command-queue. The kernel will be queued for execution on the device associated with <see cref="commandQueue"/>.</param>
        /// <param name="kernel">A valid kernel object. The OpenCL context associated with <see cref="kernel"/> and <see cref="commandQueue"/> must be the same.</param>
        /// <param name="workDimension">The number of dimensions used to specify the global work-items and work-items in the work-group.</param>
        /// <param name="globalWorkOffset">
        /// Can be used to specify an array of <see cref="workDimension"/> unsigned values that describe the offset used to calculate the global ID of a work-item. If <see cref="globalWorkOffset"/> is <c>null</c>, the global IDs start at
        /// offset (0, 0, ... 0).
        /// </param>
        /// <param name="globalWorkSize">
        /// Points to an array of <see cref="workDimension"/> unsigned values that describe the number of global work-items in <see cref="workDimension"/> dimensions that will execute the kernel function. The total number of global
        /// work-items is computed as globalWorkSize[0] *...* globalWorkSize[workDimension - 1].
        /// </param>
        /// <param name="localWorkSize">
        /// Points to an array of <see cref="workDimension"/> unsigned values that describe the number of work-items that make up a work-group (also referred to as the size of the work-group) that will execute the kernel specified by
        /// <see cref="kernel"/>. The total number of work-items in a work-group is computed as localWorkSize[0] *... * localWorkSize[workDimension - 1].
        /// </param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular kernel-instance. Event objects are unique and can be used to identify a particular kernel execution instance later on. If event is <c>null</c>, no event will be created for
        /// this kernel execution instance and therefore it will not be possible for the application to query or queue a wait for this particular kernel execution instance.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueNDRangeKernel")]
        public static extern Result EnqueueNDRangeKernel(
            [In] IntPtr commandQueue,
            [In] IntPtr kernel,
            [In] [MarshalAs(UnmanagedType.U4)] uint workDimension,
            [In] IntPtr[] globalWorkOffset,
            [In] IntPtr[] globalWorkSize,
            [In] IntPtr[] localWorkSize,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueNDRangeKernel")]
        public static extern Result EnqueueNDRangeKernel(
            [In] IntPtr commandQueue,
            [In] IntPtr kernel,
            [In] [MarshalAs(UnmanagedType.U4)] uint workDimension,
            [In] IntPtr[] globalWorkOffset,
            [In] IntPtr[] globalWorkSize,
            [In] IntPtr[] localWorkSize,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [In] IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a command to execute a native C/C++ function not compiled using the OpenCL compiler.
        /// </summary>
        /// <param name="commandQueue">
        /// A valid host command-queue. A native user function can only be executed on a command-queue created on a device that has <c>DeviceExecutionCapabilities.NativeKernel</c> capability set in <c>DeviceInformation.ExecutionCapabilities</c>
        /// return by <see cref="Device.GetDeviceInformation"/>.
        /// </param>
        /// <param name="userFunction">A pointer to a host-callable user function.</param>
        /// <param name="arguments">A pointer to the arguments list that <see cref="userFunction"/> should be called with.</param>
        /// <param name="argumentSize">The size in bytes of the arguments list that <see cref="arguments"/> points to.</param>
        /// <param name="numberOfMemoryObjects">The number of buffer objects that are passed in <see cref="arguments"/>.</param>
        /// <param name="memoryObjects">
        /// A list of valid buffer objects, if <see cref="numberOfMemoryObjects"/> is greater than 0. The buffer object values specified in <see cref="memoryObjects"/> are memory object handles returned by <see cref="CreateBuffer"/> or <c>null</c>.
        /// </param>
        /// <param name="argumentsMemoryLocation">
        /// A pointer to appropriate locations that <see cref="arguments"/> points to where memory object handles are stored. Before the user function is executed, the memory object handles are replaced by pointers to global memory.
        /// </param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular kernel-instance. Event objects are unique and can be used to identify a particular kernel execution instance later on. If event is <c>null</c>, no event will be created for
        /// this kernel execution instance and therefore it will not be possible for the application to query or queue a wait for this particular kernel execution instance.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueNativeKernel")]
        public static extern Result EnqueueNativeKernel(
            [In] IntPtr commandQueue,
            [In] IntPtr userFunction,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] arguments,
            [In] UIntPtr argumentSize,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfMemoryObjects,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] memoryObjects,
            [In] IntPtr argumentsMemoryLocation,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a marker command which waits for either a list of events to complete, or all previously enqueued commands to complete.
        /// </summary>
        /// <param name="commandQueue">A valid host command-queue.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular kernel-instance. Event objects are unique and can be used to identify a particular kernel execution instance later on. If event is <c>null</c>, no event will be created for
        /// this kernel execution instance and therefore it will not be possible for the application to query or queue a wait for this particular kernel execution instance.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 2)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueMarkerWithWaitList")]
        public static extern Result EnqueueMarkerWithWaitList(
            [In] IntPtr commandQueue,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// A synchronization point that enqueues a barrier operation.
        /// </summary>
        /// <param name="commandQueue">A valid host command queue.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular kernel-instance. Event objects are unique and can be used to identify a particular kernel execution instance later on. If event is <c>null</c>, no event will be created for
        /// this kernel execution instance and therefore it will not be possible for the application to query or queue a wait for this particular kernel execution instance.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 2)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueBarrierWithWaitList")]
        public static extern Result EnqueueBarrierWithWaitList(
            [In] IntPtr commandQueue,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a command to free the shared virtual memory allocated using clSVMAlloc or a shared system memory pointer.
        /// </summary>
        /// <param name="commandQueue">A valid host command-queue.</param>
        /// <param name="numberOfSvmPointers">The number of SVM pointers stored in <see cref="svmPointers"/>.</param>
        /// <param name="svmPointers">
        /// Specify shared virtual memory pointers to be freed. Each pointer in <see cref="svmPointers"/> that was allocated using <see cref="SvmAllocate"/> must have been allocated from the same context from which <see cref="commandQueue"/> was
        /// created. The memory associated with <see cref="svmPointers"/> can be reused or freed after the function returns.
        /// </param>
        /// <param name="svmFreePointersCallback">
        /// Specifies the callback function to be called to free the SVM pointers. <see cref="svmFreePointersCallback"/> takes four arguments: queue which is the command queue in which <see cref"EnqueueSvmFree"/> was enqueued, the count and list
        /// of SVM pointers to free and <see cref="userData"/> which is a pointer to user specified data. If <see cref="svmFreePointersCallback"/> is <c>null</c>, all pointers specified in <see cref="svmPointers"/> must be allocated using
        /// <see cref="SvmAllocate"/> and the OpenCL implementation will free these SVM pointers. <see cref="svmFreePointersCallback"/> must be a valid callback function if any SVM pointer to be freed is a shared system memory pointer i.e. not
        /// allocated using <see cref="SvmAllocate"/>. If <see cref="svmFreePointersCallback"/> is a valid callback function, the OpenCL implementation will call <see cref="svmFreePointersCallback"/> to free all the SVM pointers specified in
        /// <see cref="svmPointers"/>.
        /// </param>
        /// <param name="userData">Will be passed as the <see cref="userData"/> argument when <see cref="svmFreePointersCallback"/> is called. <see cref="userData"/> can be <c>null</c>.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular kernel-instance. Event objects are unique and can be used to identify a particular kernel execution instance later on. If event is <c>null</c>, no event will be created for
        /// this kernel execution instance and therefore it will not be possible for the application to query or queue a wait for this particular kernel execution instance.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(2, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueSVMFree")]
        public static extern Result EnqueueSvmFree(
            [In] IntPtr commandQueue,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfSvmPointers,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] svmPointers,
            [In] IntPtr svmFreePointersCallback,
            [In] IntPtr userData,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a command to do a memcpy operation.
        /// </summary>
        /// <param name="commandQueue">
        /// Refers to the host command-queue in which the read/write command will be queued. If either <see cref="destinationPointer"/> or <see cref="sourcePointer"/> is allocated using <see cref="SvmAllocate"/> then the OpenCL context allocated
        /// against must match that of <see cref="commandQueue"/>.
        /// </param>
        /// <param name="blockingCopy">
        /// Indicates if the copy operations are blocking or non-blocking.
        /// If <see cref="blockingCopy"/> is <c>true</c> (1), i.e. the copy command is blocking, <see cref="EnqueueSvmMemoryCopy"/> does not return until the buffer data has been copied into memory pointed to by <see cref="destinationPointer"/>.
        /// If <see cref="blockingCopy"/> is <c>false</c> (0), i.e. the copy command is non-blocking, <see cref="EnqueueSvmMemoryCopy"/> queues a non-blocking copy command and returns. The contents of the buffer that <see cref="destinationPointer"/>
        /// point to cannot be used until the copy command has completed. The <see cref="event"/> argument returns an event object which can be used to query the execution status of the read command. When the copy command has completed, the
        /// contents of the buffer that <see cref="destinationPointer"/> points to can be used by the application.
        /// </param>
        /// <param name="destinationPointer">The pointer to a host or SVM memory allocation where data is copied to.</param>
        /// <param name="sourcePointer">
        /// The pointer to a memory region where data is copied from. If the memory allocation(s) containing <see cref="destinationPointer"/> and/or <see cref="sourcePointer"/> are allocated using <see cref="SvmAllocate"/> and either is not
        /// allocated from the same context from which <see cref="commandQueue"/> was created the behavior is undefined.
        /// </param>
        /// <param name="size">The size in bytes of data being copied.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular kernel-instance. Event objects are unique and can be used to identify a particular kernel execution instance later on. If event is <c>null</c>, no event will be created for
        /// this kernel execution instance and therefore it will not be possible for the application to query or queue a wait for this particular kernel execution instance.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(2, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueSVMMemcpy")]
        public static extern Result EnqueueSvmMemoryCopy(
            [In] IntPtr commandQueue,
            [In] [MarshalAs(UnmanagedType.U4)] uint blockingCopy,
            [In] IntPtr destinationPointer,
            [In] IntPtr sourcePointer,
            [In] UIntPtr size,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a command to fill a region in memory with a pattern of a given pattern size.
        /// </summary>
        /// <param name="commandQueue">
        /// Refers to the host command-queue in which the fill command will be queued. The OpenCL context associated with <see cref="commandQueue"/> and SVM pointer referred to by <see cref="svmPointer"/> must be the same.
        /// </param>
        /// <param name="svmPointer">
        /// A pointer to a memory region that will be filled with <see cref="pattern"/>. It must be aligned to <see cref="patternSize"/> bytes. If <see cref="svmPointer"/> is allocated using <see cref="SvmAllocate"/> then it must be allocated
        /// from the same context from which <see cref="commandQueue"/> was created. Otherwise the behavior is undefined.
        /// </param>
        /// <param name="pattern">
        /// A pointer to the data pattern of size <see cref="patternSize"/> in bytes. pattern will be used to fill a region in <see cref="buffer"/> starting at <see cref="offset"/> and is <see cref="size"/> bytes in size. The data pattern must be
        /// a scalar or vector integer or floating-point data type. For example, if <see cref="buffer"/> is to be filled with a pattern of <c>float4</c> values, then <see cref="pattern"/> will be a pointer to a <c>float4</c> value and
        /// <see cref="patternSize"/> will be <c>sizeof(float4)</c>. The maximum value of <see cref="patternSize"/> is the size of the largest integer or floating-point vector data type supported by the OpenCL device. The memory associated with
        /// <see cref="pattern"/> can be reused or freed after the function returns.</param>
        /// <param name="patternSize">The size of <see cref="pattern"/> in bytes.</param>
        /// <param name="size">The size in bytes of region being filled starting with <see cref="svmPointer"/> and must be a multiple of <see cref="patternSize"/>.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular kernel-instance. Event objects are unique and can be used to identify a particular kernel execution instance later on. If event is <c>null</c>, no event will be created for
        /// this kernel execution instance and therefore it will not be possible for the application to query or queue a wait for this particular kernel execution instance.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(2, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueSVMMemFill")]
        public static extern Result EnqueueSvmMemoryFill(
            [In] IntPtr commandQueue,
            [In] IntPtr svmPointer,
            [In] IntPtr pattern,
            [In] UIntPtr patternSize,
            [In] UIntPtr size,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a command that will allow the host to update a region of a SVM buffer.
        /// </summary>
        /// <param name="commandQueue">Must be a valid host command-queue.</param>
        /// <param name="blockingMap">
        /// Indicates if the map operation is blocking or non-blocking.
        /// If <see cref="blockingMap"/> is <c>true</c> (1), <see cref="EnqueueSvmMap"/> does not return until the application can access the contents of the SVM region specified by <see cref="svmPointer"/> and <see cref="size"/> on the host.
        /// If <see cref="blockingMap"/> is <c>false</c> (0), i.e. map operation is non-blocking, the region specified by <see cref="svmPointer"/> and <see cref="size"/> cannot be used until the map command has completed. The <see cref="event"/>
        /// argument returns an event object which can be used to query the execution status of the map command. When the map command is completed, the application can access the contents of the region specified by <see cref="svmPointer"/> and
        /// <see cref="size"/>.
        /// </param>
        /// <param name="mapFlag">An enumeration with which determines the behavior of the map operation.</param>
        /// <param name="svmPointer">
        /// A pointer to a memory region and <see cref="size"/> in bytes that will be updated by the host. If <see cref="svmPointer"/> is allocated using <see cref="SvmAllocate"/> then it must be allocated from the same context from which
        /// <see cref="commandQueue"/> was created. Otherwise the behavior is undefined.
        /// </param>
        /// <param name="size">The size in bytes of the memory region that <see cref="svmPointer"/> points to.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular kernel-instance. Event objects are unique and can be used to identify a particular kernel execution instance later on. If event is <c>null</c>, no event will be created for
        /// this kernel execution instance and therefore it will not be possible for the application to query or queue a wait for this particular kernel execution instance.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(2, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueSVMMap")]
        public static extern Result EnqueueSvmMap(
            [In] IntPtr commandQueue,
            [In] [MarshalAs(UnmanagedType.U4)] uint blockingMap,
            [In] [MarshalAs(UnmanagedType.U8)] MapFlag mapFlag,
            [In] IntPtr svmPointer,
            [In] UIntPtr size,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a command to indicate that the host has completed updating the region given by <see cref="svmPointer"/> and which was specified in a previous call to <see cref="EnqueueSvmMap"/>.
        /// </summary>
        /// <param name="commandQueue">Must be a valid host command-queue.</param>
        /// <param name="svmPointer">
        /// A pointer that was specified in a previous call to <see cref="EnqueueSvmMap"/>. If <see cref="svmPointer"/> is allocated using <see cref="SvmAllocate"/> then it must be allocated from the same context from which
        /// <see cref="commandQueue"/> was created. Otherwise the behavior is undefined.
        /// </param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular kernel-instance. Event objects are unique and can be used to identify a particular kernel execution instance later on. If event is <c>null</c>, no event will be created for
        /// this kernel execution instance and therefore it will not be possible for the application to query or queue a wait for this particular kernel execution instance.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(2, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueSVMUnmap")]
        public static extern Result EnqueueSvmUnmap(
            [In] IntPtr commandQueue,
            [In] IntPtr svmPointer,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a command to indicate which device a set of ranges of SVM allocations should be associated with.
        /// </summary>
        /// <param name="commandQueue">A valid host command queue. The specified set of allocation ranges will be migrated to the OpenCL device associated with <see cref="commandQueue"/>.</param>
        /// <param name="numberOfSvmPointers">The number of pointers in the specified <see cref="svmPointers"/> array, and the number of sizes in the sizes array, if sizes is not <c>null</c>.</param>
        /// <param name="svmPointers"></param>
        /// <param name="sizes">A pointer to an array of pointers. Each pointer in this array must be within an allocation produced by a call to <see cref="SvmAllocate"/>.</param>
        /// <param name="memoryMigrationFlags">An enumeration that is used to specify migration options.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular kernel-instance. Event objects are unique and can be used to identify a particular kernel execution instance later on. If event is <c>null</c>, no event will be created for
        /// this kernel execution instance and therefore it will not be possible for the application to query or queue a wait for this particular kernel execution instance.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(2, 1)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueSVMMigrateMem")]
        public static extern Result EnqueueSvmMigrateMemory(
            [In] IntPtr commandQueue,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfSvmPointers,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] svmPointers,
            [In] [MarshalAs(UnmanagedType.LPArray)] UIntPtr[] sizes,
            [In] [MarshalAs(UnmanagedType.U8)] MemoryMigrationFlag memoryMigrationFlags,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        #endregion

        #region Public Deprecated Methods

        /// <summary>
        /// Enqueues a marker command.
        /// </summary>
        /// <param name="commandQueue">A valid host command queue.</param>
        /// <param name="waitEvent">The event object that serves as marker.</param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueMarker")]
        [Obsolete("This is a deprecated OpenCL 1.1 method, please use EnqueueMarkerWithWaitList instead.")]
        public static extern Result EnqueueMarker(
            [In] IntPtr commandQueue,
            [In] IntPtr waitEvent
        );

        /// <summary>
        /// Enqueues a wait for a specific event or a list of events to complete before any future commands queued in the command-queue are executed.
        /// </summary>
        /// <param name="commandQueue">A valid command-queue.</param>
        /// <param name="numberOfEventsInWaitList">Specifies the number of events given by <see cref="eventWaitList"/>.</param>
        /// <param name="eventWaitList">
        /// Events specified in <see cref="eventWaitList"/> act as synchronization points. The context associated with events in <see cref="eventWaitList"/> and <see cref="commandQueue"/> must be the same. Each event in <see cref="eventWaitList"/>
        // must be a valid event object.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueWaitForEvents")]
        [Obsolete("This is a deprecated OpenCL 1.1 method, please use EnqueueMarkerWithWaitList instead.")]
        public static extern Result EnqueueWaitForEvents(
            [In] IntPtr commandQueue,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList
        );

        /// <summary>
        /// A synchronization point that enqueues a barrier operation.
        /// </summary>
        /// <param name="commandQueue">A valid command-queue.</param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueBarrier")]
        [Obsolete("This is a deprecated OpenCL 1.1 method, please use EnqueueBarrierWithWaitList instead.")]
        public static extern Result EnqueueBarrier(
            [In] IntPtr commandQueue
        );

        /// <summary>
        /// Enqueues a command to execute a kernel on a device.
        /// </summary>
        /// <param name="commandQueue">A valid command-queue. The kernel will be queued for execution on the device associated with <see cref="commandQueue"/>.</param>
        /// <param name="kernel">A valid kernel object. The OpenCL context associated with <see cref="kernel"/> and <see cref="commandQueue"/> must be the same.</param>
        /// <param name="numberOfEventsInWaitList">The number of event in <see cref="eventWaitList"/>. If <see cref="eventWaitList"/> is <c>null</c>, then <see cref="numberOfEventsInWaitList"/ must be 0.</param>
        /// <param name="eventWaitList">
        /// Specify events that need to complete before this particular command can be executed. If <see cref="eventWaitList"/> is <c>null</c>, then this particular command does not wait on any event to complete.
        /// </param>
        /// <param name="waitEvent">
        /// Returns an event object that identifies this particular kernel-instance. Event objects are unique and can be used to identify a particular kernel execution instance later on. If event is <c>null</c>, no event will be created for
        /// this kernel execution instance and therefore it will not be possible for the application to query or queue a wait for this particular kernel execution instance.
        /// </param>
        /// <returns>Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns an error.</returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clEnqueueTask")]
        [Obsolete("This is a deprecated OpenCL 1.2 method.")]
        public static extern Result EnqueueTask(
            [In] IntPtr commandQueue,
            [In] IntPtr kernel,
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEventsInWaitList,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventWaitList,
            [Out] out IntPtr waitEvent
        );

        #endregion
    }
}