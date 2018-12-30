
#region Using Directives

using System;
using System.Runtime.InteropServices;
using OpenCl.DotNetCore.CommandQueues;
using OpenCl.DotNetCore.Interop;
using OpenCl.DotNetCore.Interop.Events;

#endregion

namespace OpenCl.DotNetCore.Events
{
    /// <summary>
    /// Represents an event, which is returned by all OpenCL methods, that take longer. They can be used await asynchronous API calls. This class is awaitable and can be used with the C# <c>await</c> keyword. Please not, that when
    /// the awaitable event is awaited, then it is auto-disposed, but when the awaitable event is not awaited, then it must be disposed of manually.
    /// </summary>
    public class AwaitableEvent : HandleBase
    {
        #region Constructors

        /// <summary>
        /// Initializes a new <see cref="AwaitableEvent"/> instance.
        /// </summary>
        /// <param name="handle">The handle to the OpenCL event.</param>
        public AwaitableEvent(IntPtr handle)
            : base(handle)
        {
            // Subscribes to the event callbacks of the OpenCL event, so that a CLR event can be raised
            EventsNativeApi.SetEventCallback(
                this.Handle,
                (int)CommandExecutionStatus.Queued,
                Marshal.GetFunctionPointerForDelegate(new AwaitableEventCallback((waitEvent, userData) => this.OnQueued?.Invoke(this, new EventArgs()))),
                IntPtr.Zero);
            EventsNativeApi.SetEventCallback(
                this.Handle,
                (int)CommandExecutionStatus.Submitted,
                Marshal.GetFunctionPointerForDelegate(new AwaitableEventCallback((waitEvent, userData) => this.OnSubmitted?.Invoke(this, new EventArgs()))),
                IntPtr.Zero);
            EventsNativeApi.SetEventCallback(
                this.Handle,
                (int)CommandExecutionStatus.Running,
                Marshal.GetFunctionPointerForDelegate(new AwaitableEventCallback((waitEvent, userData) => this.OnRunning?.Invoke(this, new EventArgs()))),
                IntPtr.Zero);
            EventsNativeApi.SetEventCallback(
                this.Handle,
                (int)CommandExecutionStatus.Complete,
                Marshal.GetFunctionPointerForDelegate(new AwaitableEventCallback((waitEvent, userData) => this.OnCompleted?.Invoke(this, new EventArgs()))),
                IntPtr.Zero);
        }

        #endregion

        #region Public Properties

        /// <summary>
        /// Gets the current command execution status code. This is the raw numeric status code, which can be helpful, when the command raised an error, to retrieve more information about the type of error that was returned.
        /// </summary>
        public int CommandExecutionStatusCode
        {
            get
            {
                return this.GetEventInformation<int>(EventInformation.CommandExecutionStatus);
            }
        }

        /// <summary>
        /// Gets the current command execution status.
        /// </summary>
        public CommandExecutionStatus CommandExecutionStatus
        {
            get
            {
                int commandExecutionStatusCode = this.CommandExecutionStatusCode;
                if (commandExecutionStatusCode >= 0)
                    return (CommandExecutionStatus)commandExecutionStatusCode;
                return CommandExecutionStatus.Error;
            }
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Retrieves the specified information about the OpenCL event.
        /// </summary>
        /// <typeparam name="T">The type of the data that is to be returned.</param>
        /// <param name="eventInformation">The kind of information that is to be retrieved.</param>
        /// <exception cref="OpenClException">If the information could not be retrieved, then an <see cref="OpenClException"/> is thrown.</exception>
        /// <returns>Returns the specified information.</returns>
        private T GetEventInformation<T>(EventInformation eventInformation)
        {
            // Retrieves the size of the return value in bytes, this is used to later get the full information
            UIntPtr returnValueSize;
            Result result = EventsNativeApi.GetEventInformation(this.Handle, eventInformation, UIntPtr.Zero, null, out returnValueSize);
            if (result != Result.Success)
                throw new OpenClException("The event information could not be retrieved.", result);
            
            // Allocates enough memory for the return value and retrieves it
            byte[] output = new byte[returnValueSize.ToUInt32()];
            result = EventsNativeApi.GetEventInformation(this.Handle, eventInformation, new UIntPtr((uint)output.Length), output, out returnValueSize);
            if (result != Result.Success)
                throw new OpenClException("The event information could not be retrieved.", result);

            // Returns the output
            return InteropConverter.To<T>(output);
        }

        #endregion

        #region Private Delegates

        /// <summary>
        /// A delegate for the callback of wait event.
        /// </summary>
        /// <param name="waitEvent">A pointer to the OpenCL event object.</param>
        /// <param name="userData">User-defined data that can be passed to the event subscription.</param>
        private delegate void AwaitableEventCallback(IntPtr waitEvent, IntPtr userData);

        #endregion

        #region Public Events

        /// <summary>
        /// An event, which is raised, when the command gets enqueued to a command-queue.
        /// </summary>
        public event EventHandler OnQueued;

        /// <summary>
        /// An event, which is raised, when the command is submitted to a device.
        /// </summary>
        public event EventHandler OnSubmitted;

        /// <summary>
        /// An event, which is raised, when the command is being executed on a device.
        /// </summary>
        public event EventHandler OnRunning;

        /// <summary>
        /// An event, which is raised, when the command completes successfully or with an error.
        /// </summary>
        public event EventHandler OnCompleted;

        #endregion

        #region IDisposable Implementation

        /// <summary>
        /// Disposes of the resources that have been acquired by the event.
        /// </summary>
        /// <param name="disposing">Determines whether managed object or managed and unmanaged resources should be disposed of.</param>
        protected override void Dispose(bool disposing)
        {
            // Checks if the event has already been disposed of, if not, then it is disposed of
            if (!this.IsDisposed)
                EventsNativeApi.ReleaseEvent(this.Handle);

            // Makes sure that the base class can execute its dispose logic
            base.Dispose(disposing);
        }

        #endregion
    }
}