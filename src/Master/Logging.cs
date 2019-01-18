// Grin Gold Miner https://github.com/urza/GrinGoldMiner
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Mozkomor.GrinGoldMiner
{
    //public static class FileInfoExtensions
    //{
    //    public static DateTime TryGetDateFromFileName(this FileInfo f, DateTime defaultValue)
    //    {
    //        try
    //        {
    //            var name = f.Name;
    //            DateTime parsed;
    //            try
    //            {
    //                if (name.Split('.').Count() > 1)
    //                {
    //                    name = name.Split('.')[0];
    //                }
    //            }
    //            catch { }

    //            if (DateTime.TryParseExact(name, "yyyyMMddHHmmss", CultureInfo.InvariantCulture, System.Globalization.DateTimeStyles.None, out parsed))
    //                return parsed;
    //            else
    //                return defaultValue;
    //        }
    //        catch
    //        {
    //            return defaultValue;
    //        }
    //    }
    //}
    public enum LogLevel
    {
        DEBUG,
        INFO,
        WARNING,
        ERROR
    }
    public enum ConsoleOutputMode
    {
        STATIC_TUI,
        ROLLING_LOG,
        WORKER_MSG
    }
    public class LogOptions
    {
        public LogLevel FileMinimumLogLevel { get; set; }
        public LogLevel ConsoleMinimumLogLevel { get; set; }

        /// <summary>
        /// How many days old logs to keep. Will delete older logs when "Logger.SetLogPath()" (once a day or on app first log write) is called.
        /// </summary>
        public int KeepDays { get; set; }

        public bool DisableLogging { get; set; }
    }
    public class Logger
    {
        private static LogOptions logOptions;
        private static string _logPath;
        private static DateTime _lastDayLogCreated;
        private static Dictionary<string, int> msgcnt = new Dictionary<string, int>();
        public static string[] last5msg = new string[5];
        public static volatile ConsoleOutputMode consoleMode = ConsoleOutputMode.STATIC_TUI;

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
            var logdir = Path.Combine(execdir, "logs");
            if (!Directory.Exists(logdir))
                Directory.CreateDirectory(logdir);
            var file = DateTime.Now.ToString("yyyyMMddHHmmss") + ".txt";
            _logPath = Path.Combine(logdir, file);
            Task.Run(() => { try { DeleteOldLogs(logdir, logOptions?.KeepDays ?? 1); } catch { } });
        }

        private static void DeleteOldLogs(string logdir, int olderThanDays)
        {
            try
            {
                if (string.IsNullOrEmpty(logdir) || olderThanDays == 0) //prevent default (0) from old configs
                    return;

                foreach (var f in Directory.GetFiles(logdir))
                {
                    var fi = new FileInfo(f);
                    if (fi.LastWriteTime < DateTime.Now.AddDays(-1 * olderThanDays))
                        fi.Delete();
                }
            }
            catch(Exception ex) { Console.WriteLine($"ERROR while deleting old logs {ex.Message}"); }
        }

        public static void SetLogOptions(LogOptions options)
        {
            logOptions = options;
        }
        public static void Log(Exception ex, [CallerFilePath]string callerFilePath = null, [CallerMemberName]string callerMemberName = null, [CallerLineNumber]int callerLineNumber = 0)
        {
            if (!string.IsNullOrEmpty(callerFilePath))
            {
                try { callerFilePath = Path.GetFileName(callerFilePath) ?? callerFilePath; } catch { }
            }

            var msg = $"Exception in: {callerFilePath} # {callerLineNumber} # {callerMemberName} Message: {ex.Message}";
            Log(LogLevel.ERROR, msg);
        }
        private static object lock1 = "";
        public static ConcurrentQueue<string> criticalErrors = new ConcurrentQueue<string>();
        public static void Log(LogLevel level, string msg)
        {
            try
            {
                if (logOptions == null)
                {
#if DEBUG
                logOptions = new LogOptions() {FileMinimumLogLevel = LogLevel.DEBUG, ConsoleMinimumLogLevel = LogLevel.DEBUG, KeepDays = 1, DisableLogging = false };
#else
                    logOptions = new LogOptions() { FileMinimumLogLevel = LogLevel.ERROR, ConsoleMinimumLogLevel = LogLevel.INFO, KeepDays = 1, DisableLogging = false };
#endif
                }

                if (level == LogLevel.ERROR && criticalErrors.Count < 1000)
                    criticalErrors.Enqueue(msg);

                if (level >= logOptions.FileMinimumLogLevel && !logOptions.DisableLogging)
                {
                    lock (lock1)
                    {
                        //prevent overloading log
                        if (level != LogLevel.ERROR || (level == LogLevel.ERROR && OverloadCheck(msg)))
                        {
                            File.AppendAllText(LogPath, $"{DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssK")}    {level.ToString()},     {msg}{Environment.NewLine}");
                        }
                    }
                }

                if (level >= logOptions.ConsoleMinimumLogLevel)
                {
                    msg = msg.Trim();
                    pushMessage($"{level.ToString(),-8}{msg}");

                    if (consoleMode == ConsoleOutputMode.ROLLING_LOG)
                    {
                        if ((level == LogLevel.ERROR) || (level == LogLevel.WARNING))
                            Console.ForegroundColor = ConsoleColor.Red;

                        Console.WriteLine($"{level.ToString(),-8}{msg}");
                        Console.ResetColor();
                    }
                }

            }
            catch { } //epic fail
        }

        private static void pushMessage(string msg)
        {
            last5msg[4] = last5msg[3];
            last5msg[3] = last5msg[2];
            last5msg[2] = last5msg[1];
            last5msg[1] = last5msg[0];
            last5msg[0] = msg;
        }

        public static string GetlastLogs()
        {
            return $"{Shorten(last5msg[0])}\n{Shorten(last5msg[1])}\n{Shorten(last5msg[2])}\n{Shorten(last5msg[3])}\n{Shorten(last5msg[4])}";
        }

        public static string Shorten(string s)
        {
            if (s == null)
                return "";

            if (s.Length > 100)
                return s.Substring(0, 100) + "...";

            return s;
        }

        //surround by lock when calling
        public static bool OverloadCheck(string msg)
        {
            if (!msgcnt.ContainsKey(msg))
            {
                msgcnt.Add(msg, 1);
                return true;
            }
            else
            {
                if (msgcnt[msg] < 60)
                {
                    var val = msgcnt[msg];
                    msgcnt[msg] = val + 1;
                    return true;
                }
                else
                {
                    return false;
                }
            }
        }
    }
}
