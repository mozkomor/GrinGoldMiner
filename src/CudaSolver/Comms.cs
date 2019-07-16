using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Runtime.Serialization.Formatters.Binary;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using SharedSerialization;

namespace CudaSolver
{
    internal class Comms
    {
        public static ConcurrentQueue<Solution> graphSolutionsOut = new ConcurrentQueue<Solution>();
        public static ConcurrentQueue<LogMessage> logsOut = new ConcurrentQueue<LogMessage>();
        public static GpuDevicesMessage gpuMsg = null;

        public static AutoResetEvent flushToMaster;
        private static TcpClient client;
        private static NetworkStream stream;
        private static Task listener;
        private static Task sender;
        private static int _port;
        public static Job nextJob = new Job();
        public static volatile bool IsTerminated = false;
        public static DateTime lastIncoming = DateTime.Now;

        public static volatile int cycleFinderTargetOverride = 0;
        public static volatile int numberOfGPUs = 0;

        static int errorCounter = 0;

        private static volatile bool IsVerified = false;
        private static string verificationMessage = null;

        internal static void ConnectToMaster(int port)
        {
            _port = port;
            flushToMaster = new AutoResetEvent(false);
            client = new TcpClient("127.0.0.1", port);
            //client = new TcpClient("10.0.0.122", port);
            if (client.Connected)
            {
                stream = client.GetStream();
                listener = Task.Factory.StartNew(() => { Listen(); }, TaskCreationOptions.LongRunning);
                sender = Task.Factory.StartNew(() => { WaitSend(); }, TaskCreationOptions.LongRunning);
            }
        }

        private static void WaitSend()
        {
            while (!IsTerminated)
            {
                try
                {
                    flushToMaster.WaitOne();

                    Solution s;
                    while (graphSolutionsOut.TryDequeue(out s))
                    {
                        if ((IsVerified && verificationMessage == null) || IsVerified)
                            (new BinaryFormatter() { AssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Full }).Serialize(stream, s);
                    }
                    if (gpuMsg != null)
                    {
                        //Console.WriteLine("sending now");
                        (new BinaryFormatter() { AssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Simple }).Serialize(stream, gpuMsg);
                        gpuMsg = null;
                        //Console.WriteLine("flushing now");
                        stream.Flush();
                    }
                    LogMessage lm;
                    while (logsOut.TryDequeue(out lm))
                    {
                        if (verificationMessage == null && lm.message != null && lm.message.ToLower().Contains("trimmed"))
                            verificationMessage = "GrinPro2.Solvers." + lm.message;

                        (new BinaryFormatter() { AssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Full }).Serialize(stream, lm);
                    }
                }
                catch (Exception ex)
                {
                    // log to local file and console
                    //Logger.Log(LogLevel.Warning, "WaitSend error", ex);
                    //Console.WriteLine("exc " + ex.InnerException.Message);
                }
            }
        }

        private static void Listen()
        {
            while (!IsTerminated)
            {
                try
                {
                    object payload = (new BinaryFormatter()).Deserialize(stream);

                    switch (payload)
                    {
                        case Job job:
                            nextJob = job;
                            lastIncoming = DateTime.Now;
                            Logger.Log(LogLevel.Debug, $"New job received: {job.pre_pow}");
                            break;
                        case GpuSettings settings:
                            if (settings.gpuSettings != null && settings.gpuSettings != "")
                            {
                                StringHelper help = new StringHelper(Encoding.ASCII.GetBytes($"{typeof(GpuSettings).ToString(),32}"));
                                var decoded = help.Decode(settings.gpuSettings);
                                if (decoded == verificationMessage)
                                    IsVerified = true;
                            }
                            else
                            {
                                cycleFinderTargetOverride = settings.targetGraphTimeOverride;
                                numberOfGPUs = settings.numberOfGPUs;
                            }
                            break;
                    }

                    errorCounter = 0;
                }
                catch (Exception ex)
                {
                    //Logger.Log(LogLevel.Warning, "Listen error! ", ex);
                    Console.WriteLine("Connection lost...");

                    Task.Delay(5);
                    try
                    {
                        while (stream.DataAvailable)
                            stream.ReadByte();
                    }
                    catch { }

                    if (errorCounter++ > 5)
                    {
                        Close();
                    }
                }
            }
        }

        internal static bool IsConnected()
        {
            return client != null && client.Connected;
        }

        internal static void SetEvent()
        {
            if (flushToMaster != null)
                flushToMaster.Set();
        }

        internal static void Close()
        {
            try
            {
                if (stream != null)
                    stream.Flush();
                Task.Delay(500).Wait();
                IsTerminated = true;
                if (stream != null)
                    stream.Close();
                if (client != null)
                    client.Close();
            }
            catch { }
        }

        private class StringHelper
        {
            private readonly Random random;
            private readonly byte[] key;
            private readonly RijndaelManaged rm;
            private readonly UTF8Encoding encoder;

            public StringHelper(byte[] input)
            {
                this.random = new Random();
                this.rm = new RijndaelManaged();
                this.encoder = new UTF8Encoding();
                this.key = input;
            }

            public string Encode(string unencrypted)
            {
                var vector = new byte[16];
                this.random.NextBytes(vector);
                var cryptogram = vector.Concat(this.Encrypt(this.encoder.GetBytes(unencrypted), vector));
                return Convert.ToBase64String(cryptogram.ToArray());
            }

            public string Decode(string encrypted)
            {
                var cryptogram = Convert.FromBase64String(encrypted);
                if (cryptogram.Length < 17)
                {
                    throw new ArgumentException("Not a valid encrypted string", "encrypted");
                }

                var vector = cryptogram.Take(16).ToArray();
                var buffer = cryptogram.Skip(16).ToArray();
                return this.encoder.GetString(this.Decrypt(buffer, vector));
            }

            private byte[] Encrypt(byte[] buffer, byte[] vector)
            {
                var encryptor = this.rm.CreateEncryptor(this.key, vector);
                return this.Transform(buffer, encryptor);
            }

            private byte[] Decrypt(byte[] buffer, byte[] vector)
            {
                var decryptor = this.rm.CreateDecryptor(this.key, vector);
                return this.Transform(buffer, decryptor);
            }

            private byte[] Transform(byte[] buffer, ICryptoTransform transform)
            {
                var stream = new MemoryStream();
                using (var cs = new CryptoStream(stream, transform, CryptoStreamMode.Write))
                {
                    cs.Write(buffer, 0, buffer.Length);
                }

                return stream.ToArray();
            }
        }
    }

    
}
