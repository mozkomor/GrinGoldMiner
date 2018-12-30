
#region Using Directives

using System;
using System.Runtime.InteropServices;

#endregion

namespace OpenCl.DotNetCore.Interop.Events
{
    /// <summary>
    /// Represents a wrapper for the native methods of the OpenCL Events API.
    /// </summary>
    public static class EventsNativeApi
    {
        #region Public Static Methods

        /// <summary>
        /// Waits on the host thread for commands identified by event objects in <see cref="eventList"/> to complete. A command is considered complete if its execution status is <c>CommandExecutionState.Complete</c> or a negative value. The
        /// events specified in <see cref="eventList"/> act as synchronization points.
        /// </summary>
        /// <param name="numberOfEvents">The number of event contained in <see cref="eventList"/>.</param>
        /// <param name="eventList">A list of event on which is to be waited on the host thread.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the execution status of all events in <see cref="eventList"/> is <c>CommandExecutionState.Complete</c>. Otherwise, it returns one of the following errors:
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="numberOfEvents"/> is zero or <see cref="eventList"/> is <c>null</c>.
        /// 
        /// <c>Result.InvalidContext</c> if events specified in <see cref="eventList"/> do not belong to the same context.
        /// 
        /// <c>Result.InvalidEvent</c> if event objects specified in <see cref="eventList"/> are not valid event objects.
        /// 
        /// <c>Result.ExecutionStatusErrorForEventsInWaitList</c> if the execution status of any of the events in <see cref="eventList"/> is a negative integer value.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clWaitForEvents")]
        public static extern Result WaitForEvents(
            [In] [MarshalAs(UnmanagedType.U4)] uint numberOfEvents,
            [In] [MarshalAs(UnmanagedType.LPArray)] IntPtr[] eventList
        );

        /// <summary>
        /// Returns information about the event object.
        /// </summary>
        /// <param name="eventPointer">Specifies the event object being queried./param>
        /// <param name="parameterName">An enumeration constant that identifies the event information being queried.</param>
        /// <param name="parameterValueSize">Specifies the size in bytes of memory pointed to by <see cref="parameterValue"/>. This size in bytes must be greater than or equal to the size of return type specified.</param>
        /// <param name="parameterValue">A pointer to memory location where appropriate values for a given <see cref="parameterName"/>. If <see cref="parameterValue"/> is <c>null</c>, it is ignored.</param>
        /// <param name="parameterValueSizeReturned">Returns the actual size in bytes of data being queried by <see cref="parameterValue"/>. If <see cref="parameterValueSizeReturned"/> is <c>null</c>, it is ignored.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following:
        /// 
        /// <c>Result.InvalidEvent</c> if <see cref="event"/> is not a valid event object.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="parameterName"/> is not one of the supported values or if size in bytes specified by <see cref="parameterValueSize"/> is less than size of return type and <see cref="parameterValue"/> is
        /// not a <c>null</c> value or if information to query given in <see cref="parameterName"/> cannot be queried for event.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clGetEventInfo")]
        public static extern Result GetEventInformation(
            [In] IntPtr eventPointer,
            [In] [MarshalAs(UnmanagedType.U4)] EventInformation parameterName,
            [In] UIntPtr parameterValueSize,
            [Out] byte[] parameterValue,
            [Out] out UIntPtr parameterValueSizeReturned
        );

        /// <summary>
        /// Creates a user event object. User events allow applications to enqueue commands that wait on a user event to finish before the command is executed by the device. The execution status of the user event object created is set to
        /// <c>CommandExecutionStatus.Submitted</c>.
        /// </summary>
        /// <param name="context">A valid OpenCL context.</param>
        /// <param name="errorCode">Returns an appropriate error code. If <see cref="errorCode"/> is <c>null</c>, no error code is returned.</param>
        /// <returns>
        /// Returns a valid non-zero event object and <see cref="errorCode"/> is set to <c>CommandExecutionStatus.Complete<c> if the user event object is created successfully. Otherwise, it returns a <c>null</c> value with one of the following
        /// error values returned in <see cref="errorCode"/:
        /// 
        /// <c>Result.InvalidContext</c> if <see cref="context"/> is not a valid context.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 1)]
        [DllImport("OpenCL", EntryPoint = "clCreateUserEvent")]
        public static extern IntPtr CreateUserEvent(
            [In] IntPtr context,
            [Out] [MarshalAs(UnmanagedType.I4)] out Result errorCode
        );

        /// <summary>
        /// Increments the event reference count. The OpenCL commands that return an event perform an implicit retain.
        /// </summary>
        /// <param name="eventPointer">Event object being retained.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following:
        /// 
        /// <c>Result.InvalidEvent</c> if <see cref="event"/> is not a valid event object.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clRetainEvent")]
        public static extern Result RetainEvent(
            [In] IntPtr eventPointer
        );

        /// <summary>
        /// Decrements the event reference count. The event object is deleted once the reference count becomes zero, the specific command identified by this event has completed (or terminated) and there are no commands in the command-queues of
        /// a context that require a wait for this event to complete. Using this function to release a reference that was not obtained by creating the object or by calling <see cref="RetainEvent"/> causes undefined behavior. Developers should
        /// be careful when releasing their last reference count on events created by <see cref="CreateUserEvent"/> that have not yet been set to status of <c>CommandExecutionStatus.Complete</c> or an error. If the user event was used in the
        /// <see cref="eventWaitList"/> argument passed to a Enqueue*** API or another application host thread is waiting for it in <see cref="WaitForEvents"/>, those commands and host threads will continue to wait for the event status to reach
        /// <c>CommandExecutionStatus.Complete</c> or error, even after the user has released the object. Since in this scenario the developer has released his last reference count to the user event, it would be in principle no longer valid for
        /// him to change the status of the event to unblock all the other machinery. As a result the waiting tasks will wait forever, and associated events, memory objects, command queues and contexts are likely to leak. In-order command queues
        /// caught up in this deadlock may cease to do any work.
        /// </summary>
        /// <param name="eventPointer">Event object being released.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following:
        /// 
        /// <c>Result.InvalidEvent</c> if <see cref="event"/> is not a valid event object.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 0)]
        [DllImport("OpenCL", EntryPoint = "clReleaseEvent")]
        public static extern Result ReleaseEvent(
            [In] IntPtr eventPointer
        );

        /// <summary>
        /// Sets the execution status of a user event object. If there are enqueued commands with user events in the <see cref="eventWaitList"/> argument of Enqueue*** commands, the user must ensure that the status of these user events being waited
        /// on are set using <see cref="SetUserEventStatus"/> before any OpenCL APIs that release OpenCL objects except for event objects are called, otherwise the behavior is undefined.
        /// </summary>
        /// <param name="eventPointer">A user event object created using <see cref="CreateUserEvent"/>.</param>
        /// <param name="executionStatus">
        /// Specifies the new execution status to be set and can be <c>CommandExecutionStatus.Complete</c> or a negative integer value to indicate an error. A negative integer value causes all enqueued commands that wait on this user event to be
        /// terminated. <see cref="SetUserEventStatus"/> can only be called once to change the execution status of event.
        /// </param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following:
        /// 
        /// <c>Result.InvalidEvent</c> if <see cref="event"/> is not a valid user event object.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="executionStatus"/> is not <c>CommandExecutionStatus.Complete</c> or a negative integer value.
        /// 
        /// <c>Result.InvalidOperation</c> if the <see cref="executionStatus"/> for <see cref="event"/> has already been changed by a previous call to <see cref="SetUserEventStatus"/>.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 1)]
        [DllImport("OpenCL", EntryPoint = "clSetUserEventStatus")]
        public static extern Result SetUserEventStatus(
            [In] IntPtr eventPointer,
            [In] [MarshalAs(UnmanagedType.I4)] int executionStatus
        );

        /// <summary>
        /// Registers a user callback function for a specific command execution status.
        /// </summary>
        /// <param name="eventPointer">A valid event object.</param>
        /// <param name="commandExecutionCallbackType">
        /// Specifies the command execution status for which the callback is registered. The command execution callback values for which a callback can be registered are <c>CommandExecutionStatus.Submitted</c>,
        /// <c>CommandExecutionStatus.Running</c>, or <c>CommandExecutionStatus.Complete</c>. There is no guarantee that the callback functions registered for various execution status values for an event will be called in the exact order that
        /// the execution status of a command changes. Furthermore, it should be noted that receiving a call back for an event with a status other than <c>CommandExecutionStatus.Complete</c>, in no way implies that the memory model or execution
        /// model as defined by the OpenCL specification has changed. For example, it is not valid to assume that a corresponding memory transfer has completed unless the event is in a state <c>CommandExecutionStatus.Complete</c>. The callback
        /// function registered for a <see cref="commandExecutionCallbackType"/> value of <c>CommandExecutionStatus.Complete</c> will be called when the command has completed successfully or is abnormally terminated.
        /// </param>
        /// <param name="notificationCallback">
        /// The event callback function that can be registered by the application. This callback function may be called asynchronously by the OpenCL implementation. It is the application's responsibility to ensure that the callback function is
        /// thread-safe. The parameters to this callback function are:
        /// 
        /// <see cref="event"/> is the event object for which the callback function is invoked.
        /// 
        /// <see cref="eventCommandExecutionStatus"/> represents the execution status of command for which this callback function is invoked. If the callback is called as the result of the command associated with event being abnormally terminated, an appropriate error code for
        /// the error that caused the termination will be passed to <see cref="eventCommandExecutionStatus"/> instead.
        /// 
        /// <see cref="userData"/> is a pointer to user supplied data.
        /// </param>
        /// <param name="userData">Will be passed as the <see cref="userData"/> argument when <see cref="notificationCallback"/> is called. <see cref="userData"/> can be <c>null</c>.</param>
        /// <returns>
        /// Returns <c>Result.Success</c> if the function is executed successfully. Otherwise, it returns the following:
        /// 
        /// <c>Result.InvalidEvent</c> if <see cref="event"/> is not a valid event object.
        /// 
        /// <c>Result.InvalidValue</c> if <see cref="notificationCallback"/> is <c>null</c> or if <see cref="commandExecutionCallbackType"/> is not <c>CommandExecutionStatus.Complete</c>.
        /// 
        /// <c>Result.OutOfResources</c> if there is a failure to allocate resources required by the OpenCL implementation on the device.
        /// 
        /// <c>Result.OutOfHostMemory</c> if there is a failure to allocate resources required by the OpenCL implementation on the host.
        /// </returns>
        [IntroducedInOpenCl(1, 1)]
        [DllImport("OpenCL", EntryPoint = "clSetEventCallback")]
        public static extern Result SetEventCallback(
            [In] IntPtr eventPointer,
            [In] [MarshalAs(UnmanagedType.I4)] int commandExecutionCallbackType,
            [In] IntPtr notificationCallback,
            [In] IntPtr userData
        );
        
        #endregion
    }
}