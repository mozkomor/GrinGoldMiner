// Grin Gold Miner https://github.com/urza/GrinGoldMiner
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text;

namespace Mozkomor.GrinGoldMiner
{
    public enum LogLevel
    {
        DEBUG,
        INFO,
        WARNING,
        ERROR
    }
    public class LogOptions
    {
        public LogLevel FileMinimumLogLevel { get; set; }
        public LogLevel ConsoleMinimumLogLevel { get; set; }
    }
    public class Logger
    {
        private static LogOptions logOptions;
        private static string _logPath;
        private static DateTime _lastDayLogCreated;
        private static string LogPath
        {
            get
            {
                if (string.IsNullOrEmpty(_logPath))
                {
                    SetLogPath();
                    return _logPath;
                }
                else
                {
                    if (File.Exists(_logPath) && DateTime.Today > _lastDayLogCreated)
                    {
                        SetLogPath();
                    }
                    return _logPath;
                }
            }
        }
        private static void SetLogPath()
        {
            _lastDayLogCreated = DateTime.Today;
            var execdir = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            var logdir = Path.Combine(execdir, "logs/");
            if (!Directory.Exists(logdir))
                Directory.CreateDirectory(logdir);
            var file = DateTime.Now.ToString("yyyyMMddHHmmss")+".txt";
            _logPath = Path.Combine(logdir, file);
        }
     
        public static void SetLogOptions(LogOptions options)
        {
            logOptions = options;
        }
        public static void Log(Exception ex, [CallerFilePath]string callerFilePath = null, [CallerMemberName]string callerMemberName = null, [CallerLineNumber]int callerLineNumber = 0)
        {
            var msg = $"Exception in # File: {callerFilePath} # Line: {callerLineNumber} # Member: {callerMemberName} Message: {ex.Message}";
            Log(LogLevel.ERROR, msg);
        }
        private static object lock1 = "";
        public static void Log(LogLevel level, string msg)
        {
            if (logOptions == null)
            {
#if DEBUG
                logOptions = new LogOptions() { FileMinimumLogLevel = LogLevel.DEBUG, ConsoleMinimumLogLevel = LogLevel.DEBUG };
#else
                logOptions = new LogOptions() { FileMinimumLogLevel = LogLevel.INFO, ConsoleMinimumLogLevel = LogLevel.INFO };
#endif
            }

            if (level >= logOptions.FileMinimumLogLevel)
                lock(lock1)
                    File.AppendAllText(LogPath, $"{DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssK")}    {level.ToString()},     {msg}{Environment.NewLine}");

            if (level >= logOptions.ConsoleMinimumLogLevel)
            {
                if ((level == LogLevel.ERROR) || (level == LogLevel.WARNING))
                    Console.ForegroundColor = ConsoleColor.Red;

                Console.WriteLine($"{level.ToString()}    {msg}");
                Console.ResetColor();
            }
        }
    }
}
