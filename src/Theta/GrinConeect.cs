// Cuckoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon
// This management part of Theta optimized miner is covered by the FAIR MINING license


using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Theta
{
    /// <summary>
    /// Hey, hey, heeeey
    /// </summary>
    class GrinConeeect
    {
        string _ip;
        int _port;
        public bool IsConnected;
        private TcpClient client;
        private NetworkStream stream;
        private StreamReader reader;
        private Task watchdog = null;
        private Task listener = null;
        private CancellationTokenSource wdSource = new CancellationTokenSource();
        private CancellationTokenSource listenerSource = new CancellationTokenSource();
        private CancellationToken wdCancel;
        private CancellationToken listenerCancel;
        public JobTemplate CurrentJob = null;

        public DateTime lastComm = DateTime.Now;
        public volatile bool WaitForJob = false;
        public int BadPacketCnt = 0;

        public GrinConeeect(string ip, int port)
        {
            _ip = ip;
            _port = port;

            wdCancel = wdSource.Token;
            listenerCancel = listenerSource.Token;

            Connect();
        }

        public bool GrinSend<T>(T message)
        {
            try
            {
                string output = JsonConvert.SerializeObject(message, Formatting.None, new JsonSerializerSettings() { NullValueHandling = NullValueHandling.Ignore });

                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("TCP OUT: " + output + Environment.NewLine);
                Console.ResetColor();

                byte[] bmsg = Encoding.UTF8.GetBytes(output+"\n");
                stream.Write(bmsg, 0, bmsg.Length);
                stream.FlushAsync();

                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return false;
            }
        }

        public void GrinClose()
        {
            try
            {
                if (client != null && client.Connected)
                {
                    stream.Close();
                    client.Close();
                }
                if (watchdog != null)
                    wdSource.Cancel();
                if (listener != null)
                    listenerSource.Cancel();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }

        private void Connect()
        {
            try
            {
                client = new TcpClient(_ip, _port);

                if (client.Connected)
                {
                    IsConnected = true;
                    stream = client.GetStream();
                    reader = new StreamReader(stream);

                    if (watchdog != null)
                        watchdog = Task.Factory.StartNew(() => { Monitor(); }, wdCancel);
                    listener = Task.Factory.StartNew(() => { Listen(); }, listenerCancel);
                }
                else
                    IsConnected = false;

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }

        private void Listen()
        {
            try
            {
                while (client.Connected)
                {
                    if ((DateTime.Now - lastComm) > TimeSpan.FromMinutes(30) || BadPacketCnt > 10)
                        break;

                    string message = reader.ReadLine();

                    if (string.IsNullOrEmpty(message) || string.IsNullOrWhiteSpace(message))
                    {
                        Console.WriteLine("Bad TCP packet!");
                        BadPacketCnt++;
                        continue;
                    }

                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine();
                    Console.WriteLine("TCP IN: " + message + Environment.NewLine);
                    Console.ResetColor();




                    try
                    {
                        JObject msg = JsonConvert.DeserializeObject(message, new JsonSerializerSettings() { NullValueHandling = NullValueHandling.Ignore }) as JObject;

                        var method = ((JObject)msg)["method"].ToString();
                        string para = "";
                        if (msg.ContainsKey("params"))
                            para = msg["params"].ToString()/*.Replace("\\", "")*/;

                        BadPacketCnt = 0;

                        switch (method)
                        {
                            case "job":
                                 CurrentJob = JsonConvert.DeserializeObject<JobTemplate>(para, new JsonSerializerSettings() { NullValueHandling = NullValueHandling.Ignore });
                                WaitForJob = false;
                                lastComm = DateTime.Now;
                                break;
                            case "submit":
                                if (msg.ContainsKey("result") && msg["result"].ToString() == "ok")
                                {
                                    Console.ForegroundColor = ConsoleColor.Cyan;
                                    Console.WriteLine(
@"
--------
ACCEPTED
--------
"
                                        );
                                    Console.ResetColor();
                                }
                                break;
                            default:
                                Console.WriteLine(para);
                                break;
                        }


                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine(ex.Message);
                    }
                }
                IsConnected = false;
                // TODO REMOVE when recconect is added
                WaitForJob = false;
                Console.WriteLine("Connection dropped.");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }


        private void Monitor()
        {
            while (!watchdog.IsCanceled)
            {


                Task.Delay(1000).Wait();
            }
        }

        internal void SendSolution(Solution activeSolution, List<uint> sols)
        {
            try
            {
                SubmitParams pow = new SubmitParams() { height = activeSolution.height, nonce = activeSolution.nonce, pow = sols };
                GrinRpcRequest request = new GrinRpcRequest(GrinCommand.Solution);
                request.SetParams(pow);

                if (GrinSend<GrinRpcRequest>(request))
                {

                }
            }
            catch { }
        }
    }

    public class GrinRpcRequest
    {
        public string jsonrpc = "2.0";
        public string method = "status";
        public string id = "Stratum";
        public object @params = null;

        public GrinRpcRequest(GrinCommand cmd)
        {
            switch (cmd)
            {
                case GrinCommand.Status:
                    method = "status";
                    break;
                case GrinCommand.Solution:
                    method = "submit";
                    break;
            }
        }

        public void SetParams<T>(T param)
        {
            @params = param;
            //@params = JsonConvert.SerializeObject(param, Formatting.None, new JsonSerializerSettings() { NullValueHandling = NullValueHandling.Ignore });
        }
    }

    public enum GrinCommand
    {
        Status,
        Solution,
        Login,
        GetJob
    }

    public class SubmitParams
    {
        public UInt64 height;
	    public UInt64 nonce;
        public List<UInt32> pow;
    }

    public class JobTemplate
    {
        public Int64 height;
        public Int64 difficulty;
        public string pre_pow;

        public byte[] GetHeader()
        {
            return Enumerable.Range(0, pre_pow.Length)
                     .Where(x => x % 2 == 0)
                     .Select(x => Convert.ToByte(pre_pow.Substring(x, 2), 16))
                     .ToArray();
        }
    }
}
