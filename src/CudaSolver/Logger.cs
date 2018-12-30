using System;
using System.Collections.Generic;
using System.Text;

namespace CudaSolver
{
    public static class Logger
    {
        public static void Log(LogLevel level, string message, Exception e = null)
        {
            // error message overload detection with a dictionary...
        }
    }

    public enum LogLevel
    {
        Debug,
        Info,
        Warning,
        Error
    }
}
