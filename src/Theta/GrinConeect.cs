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
        private int attempts;
        private bool terminated = false;
        private int mined = 0;
        internal Stats statistics;

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
                terminated = true;

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
                if (client != null)
                    client.Dispose();

                Console.WriteLine("Connecting to : " + _ip);

                client = new TcpClient(_ip, _port);

                if (client.Connected)
                {
                    Console.WriteLine("Connected to node.");

                    BadPacketCnt = 0;
                    attempts = 0;
                    IsConnected = true;
                    stream = client.GetStream();
                    reader = new StreamReader(stream);

                    if (watchdog == null)
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
                            case "getjobtemplate":
                                if (msg.ContainsKey("result"))
                                {
                                    CurrentJob = JsonConvert.DeserializeObject<JobTemplate>(msg["result"].ToString(), new JsonSerializerSettings() { NullValueHandling = NullValueHandling.Ignore });
                                    if (CurrentJob != null && CurrentJob.pre_pow != null && CurrentJob.pre_pow != "")
                                    {
                                        WaitForJob = false;
                                        lastComm = DateTime.Now;
                                    }
                                }
                                break;
                            case "job":
                                 CurrentJob = JsonConvert.DeserializeObject<JobTemplate>(para, new JsonSerializerSettings() { NullValueHandling = NullValueHandling.Ignore });
                                WaitForJob = false;
                                lastComm = DateTime.Now;
                                break;
                            case "submit":
                                if (msg.ContainsKey("result") && msg["result"].ToString() == "ok")
                                {
                                    Console.ForegroundColor = ConsoleColor.Cyan;
                                    Console.WriteLine("Share accepted");
                                    Console.ResetColor();
                                }
                                else if (msg.ContainsKey("result") && msg["result"].ToString().StartsWith("blockfound"))
                                {
                                    Console.ForegroundColor = ConsoleColor.Cyan;
                                    Console.WriteLine("###################################");
                                    Console.WriteLine("######  Block mined!  #" + (++mined).ToString("D4") + "  ######"); // 8 chars
                                    Console.WriteLine("###################################");
                                    Console.ResetColor();
                                    statistics.mined++;
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
                // reconnect
                if (!terminated)
                {
                    Console.WriteLine("Reconnecting");
                    Connect();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }


        private void Monitor()
        {
            attempts = 0;
            Task.Delay(5000).Wait();
            while (!watchdog.IsCanceled)
            {
                try
                {
                    if (!client.Connected)
                    {
                        Console.WriteLine("Reconnecting");
                        Connect();
                        attempts++;
                    }
                }
                catch { }

                Task.Delay(2000).Wait();
            }
        }

        internal void SendSolution(Solution activeSolution, List<uint> sols)
        {
            try
            {
                SubmitParams pow = new SubmitParams() { height = activeSolution.height, nonce = activeSolution.nonce, pow = sols, job_id = activeSolution.jobId };
                GrinRpcRequest request = new GrinRpcRequest(GrinCommand.Solution);
                request.SetParams(pow);

                if (GrinSend<GrinRpcRequest>(request))
                {

                }
            }
            catch { }
        }

        public void SendLogin(string username, string password)
        {
            try
            {
                LoginParams lp = new LoginParams() { login = username, pass = password };
                GrinRpcRequest request = new GrinRpcRequest(GrinCommand.Login);
                request.SetParams(lp);

                if (GrinSend<GrinRpcRequest>(request))
                {
                    Console.WriteLine("Login sent.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
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
                case GrinCommand.Login:
                    method = "login";
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

       /*
   let params_in = types::SubmitParams {
    height: height,
    job_id: job_id,
    edge_bits: edge_bits,
    nonce: nonce,
    pow: pow,
   };
       */
    public class SubmitParams
    {
        public UInt64 height;
        public UInt64 job_id;
        public UInt32 edge_bits = 29;
        public UInt64 nonce;
        public List<UInt32> pow;
    }

    public class LoginParams
    {
        public string login;
        public string pass;
        public string agent = "mozkomor";
    }

    public class JobTemplate
    {
        public UInt64 height;
        public UInt64 job_id;
        public UInt64 difficulty;
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
