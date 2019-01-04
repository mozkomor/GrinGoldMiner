// Grin Gold Miner https://github.com/urza/GrinGoldMiner
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon

using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using SharedSerialization;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Security;
using System.Net.Sockets;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Mozkomor.GrinGoldMiner
{
    /// <summary>
    /// Hey, hey, heeeey
    /// </summary>
    class StratumConnet
    {
        string ip;
        int port;
        public string login;
        public string password;
        public byte id;
        public bool notifyWorkers = false;

        /// <summary>
        /// is listening to TCP client in listener loop
        /// </summary>
        public volatile bool IsConnected;

        private TcpClient client;
        private NetworkStream stream;
        private SslStream streamTLS;
        private StreamReader reader;
        private Task watchdog = null;
        private Task listener = null;
        //private CancellationTokenSource wdSource = new CancellationTokenSource();
        //private CancellationTokenSource listenerSource = new CancellationTokenSource();
        //private CancellationToken wdCancel;
        //private CancellationToken listenerCancel;
        public Job CurrentJob = null;
        public Job PrevJob = null;

        public DateTime lastComm = DateTime.Now;
        public int BadPacketCnt = 0;
        private volatile bool terminated = false;
        private int mined = 0;
        internal Stats statistics;

        public Action ReconnectAction { get; internal set; }

        public StratumConnet(string _ip, int _port, byte _id, string _login, string _pwd)
        {
            this.ip = _ip;
            this.port = _port;
            id = _id;
            login = _login;
            password = _pwd;

            //wdCancel = wdSource.Token;
            //listenerCancel = listenerSource.Token;

            IsConnected = false;
        }

       

        //connect and ask for job
        public void Connect()
        {
            try
            {
                if (client != null)
                    client.Dispose();

                System.Net.ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12 | SecurityProtocolType.Tls11 | SecurityProtocolType.Tls;

                Logger.Log(LogLevel.INFO, "Connecting to : " + ip);

                client = new TcpClient(ip, port);

                if (client.Connected)
                {
                    Logger.Log(LogLevel.INFO, "Connected to node.");

                    BadPacketCnt = 0;
                    IsConnected = true;
                    terminated = false;
                    if (port < 23416)
                    {
                        stream = client.GetStream();
                        reader = new StreamReader(stream);
                    }
                    else
                    {
                        streamTLS = new SslStream(client.GetStream(), false, new RemoteCertificateValidationCallback(ValidateServerCertificate), null);
                        streamTLS.AuthenticateAsClient(ip);
                        reader = new StreamReader(streamTLS);
                    }                    

                    if (watchdog == null)
                        watchdog = Task.Factory.StartNew(() => { DisconnectMonitor(); }, TaskCreationOptions.LongRunning);
                        //watchdog = Task.Factory.StartNew(() => { DisconnectMonitor(); }, wdCancel, TaskCreationOptions.LongRunning, TaskScheduler.Default);

                    if (listener == null)
                        listener = Task.Factory.StartNew(() => { Listen(); }, TaskCreationOptions.LongRunning);
                    //listener = Task.Factory.StartNew(() => { Listen(); }, listenerCancel, TaskCreationOptions.LongRunning, TaskScheduler.Default);

                }
                else
                    IsConnected = false;

            }
            catch (Exception ex)
            {
                IsConnected = false;
                Logger.Log(ex);
            }
        }

        //tls
        private bool ValidateServerCertificate(object sender, X509Certificate certificate, X509Chain chain, SslPolicyErrors sslPolicyErrors)
        {
            if ((sslPolicyErrors == SslPolicyErrors.None) || (sslPolicyErrors == SslPolicyErrors.RemoteCertificateNameMismatch))
                return true;

            Logger.Log(LogLevel.ERROR, $"Certificate error: {sslPolicyErrors}");

            // Do not allow this client to communicate with unauthenticated servers.
            return false;
        }

        //should be started as long running
        //waiting for messages from stream and reading them
        public  void Listen()
        {
            Logger.Log(LogLevel.DEBUG, $"begin listen for sc id {id} from thread {Environment.CurrentManagedThreadId}");
            try
            {
                while (client.Connected)
                {
                    if (terminated)
                        break;

                    if ((DateTime.Now - lastComm) > TimeSpan.FromMinutes(30) || BadPacketCnt > 10)
                        break;

                    string message = reader.ReadLine();

                    if (string.IsNullOrWhiteSpace(message))
                    {
                        Logger.Log(LogLevel.DEBUG, "Epmty read from reader");
                        BadPacketCnt++;
                        continue;
                    }

                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine();
                    Logger.Log(LogLevel.INFO, "TCP IN: " + message + Environment.NewLine);
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
                                    PrevJob = CurrentJob ?? null;
                                    CurrentJob = new Job(JsonConvert.DeserializeObject<JobTemplate>(msg["result"].ToString(), new JsonSerializerSettings() { NullValueHandling = NullValueHandling.Ignore }));
                                    if (CurrentJob != null && CurrentJob.pre_pow != null && CurrentJob.pre_pow != "")
                                    {
                                        lastComm = DateTime.Now;
                                        if (notifyWorkers)
                                            WorkerManager.newJobReceived(CurrentJob);
                                    }
                                }
                                break;
                            case "job":
                                PrevJob = CurrentJob ?? null;
                                CurrentJob = new Job(JsonConvert.DeserializeObject<JobTemplate>(para, new JsonSerializerSettings() { NullValueHandling = NullValueHandling.Ignore }));
                                if (CurrentJob != null && CurrentJob.pre_pow != null && CurrentJob.pre_pow != "")
                                {
                                    lastComm = DateTime.Now;
                                    if (notifyWorkers)
                                        WorkerManager.newJobReceived(CurrentJob);
                                }
                                break;
                            case "submit":
                                if (msg.ContainsKey("result") && msg["result"].ToString() == "ok")
                                {
                                    Console.ForegroundColor = ConsoleColor.Cyan;
                                    Logger.Log(LogLevel.INFO, "Share accepted");
                                    Console.ResetColor();
                                }
                                else if (msg.ContainsKey("result") && msg["result"].ToString().StartsWith("blockfound"))
                                {
                                    Console.ForegroundColor = ConsoleColor.Cyan;
                                    Console.WriteLine("###################################");
                                    Logger.Log(LogLevel.INFO, "######  Block mined!  #" + (++mined).ToString("D4") + "  ######"); // 8 chars
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
                    catch (System.IO.IOException)
                    { }
                    catch (System.Net.Sockets.SocketException)
                    { }
                    catch (Exception ex)
                    {
                        Logger.Log(ex);
                    }
                }
                IsConnected = false; //more like "IsListening" 
                Logger.Log(LogLevel.DEBUG, $"Listener dropped for stratum connection id {id} on thread {Environment.CurrentManagedThreadId}");
                //listenerCancel.ThrowIfCancellationRequested();

            }
            catch (Exception ex)
            {
                IsConnected = false;
                Logger.Log(ex);
            }
        }

        //close connections
        public void StratumClose()
        {
            try
            {
                terminated = true;

                if (client != null && client.Connected)
                {
                    if (stream != null)
                    {
                        stream.Close();
                        stream.Dispose();
                    }
                    if (streamTLS != null)
                    {
                        streamTLS.Close();
                        streamTLS.Dispose();
                    }
                    client.Close();
                    client.Dispose();
                }
                //if (watchdog != null)
                //    wdSource.Cancel();
                //if (listener != null)
                //    listenerSource.Cancel();
                listener = null;
                //watchdog = null;

                Logger.Log(LogLevel.DEBUG, $"Closed connection id {id}");
            }
            catch (Exception ex)
            {
                Logger.Log(ex);
            }
        }

        //watchdog, if we are disconnected, try to reconnect
        public void DisconnectMonitor()
        {
            Task.Delay(5000).Wait();
            while (!terminated)
            {
                try
                {
                    if (!IsConnected) //if (!client.Connected)
                    {
                        Logger.Log(LogLevel.DEBUG, $"Reconnecting from DisconnectMonitor, SC ID {id}");

                        
                        //wdSource.Cancel();
                        //listenerSource.Cancel();
                        //wdSource.Dispose();
                        //listenerSource.Dispose();
                        StratumClose();
                        
                        ReconnectAction();
                    }
                }
                catch (Exception ex) { Logger.Log(ex); }

                Task.Delay(2000).Wait();
            }
        }

        //Send serialized class into tcp connection
        //login, getjob...
        public bool GrinSend<T>(T message)
        {
            try
            {
                string output = JsonConvert.SerializeObject(message, Formatting.None, new JsonSerializerSettings() { NullValueHandling = NullValueHandling.Ignore });

                Console.ForegroundColor = ConsoleColor.Green;
                Logger.Log(LogLevel.INFO, "TCP OUT: " + output + Environment.NewLine);
                Console.ResetColor();

                byte[] bmsg = Encoding.UTF8.GetBytes(output + "\n");

                if (streamTLS != null)
                {
                    streamTLS.Write(bmsg, 0, bmsg.Length);
                    streamTLS.FlushAsync();
                }
                else
                {
                    stream.Write(bmsg, 0, bmsg.Length);
                    stream.FlushAsync();
                }

                return true;
            }
            catch (Exception ex)
            {
                Logger.Log(ex);
                return false;
            }
        }

        internal void SendSolution(Solution activeSolution)
        {
            try
            {
                SubmitParams pow = new SubmitParams() { height = activeSolution.job.height, nonce = activeSolution.job.nonce, pow = activeSolution.nonces.ToList(), job_id = activeSolution.job.jobID };
                StratumRpcRequest request = new StratumRpcRequest(StratumCommand.Solution);
                request.SetParams(pow);

                if (GrinSend<StratumRpcRequest>(request))
                {

                }
            }
            catch(Exception ex) { Logger.Log(ex); }
        }

        internal void KeepAlive()
        {
            try
            {
                StratumRpcRequest request = new StratumRpcRequest(StratumCommand.Keepalive);

                if (GrinSend<StratumRpcRequest>(request))
                {
                    Logger.Log(LogLevel.DEBUG, $"keepalive sent for connection id {id}");
                }
            }
            catch (Exception ex)
            {
                Logger.Log(ex);

            }
        }

        public void SendLogin()
        {
            try
            {
                LoginParams lp = new LoginParams() { login = this.login, pass = this.password };
                StratumRpcRequest request = new StratumRpcRequest(StratumCommand.Login);
                request.SetParams(lp);

                if (GrinSend<StratumRpcRequest>(request))
                {
                    Logger.Log(LogLevel.DEBUG, $"Login sent for connection id {id}.");
                }
            }
            catch (Exception ex)
            {
                Logger.Log(ex);

            }
        }

        public void RequestJob()
        {
            try
            {
                StratumRpcRequest request = new StratumRpcRequest(StratumCommand.GetJob);

                if (GrinSend<StratumRpcRequest>(request))
                {
                    Logger.Log(LogLevel.DEBUG, $"job request sent for connection id {id}");
                }
            }
            catch (Exception ex)
            {
                Logger.Log(ex);
            }
        }

        public override string ToString()
        {
            return $"{ip}:{port} IsConnected:{IsConnected} id:{id}";
        }

    }

    public class StratumRpcRequest
    {
        public string jsonrpc = "2.0";
        public string method = "status";
        public string id = "Stratum";
        public object @params = null;

        public StratumRpcRequest(StratumCommand cmd)
        {
            switch (cmd)
            {
                case StratumCommand.Status:
                    method = "status";
                    break;
                case StratumCommand.Solution:
                    method = "submit";
                    break;
                case StratumCommand.Login:
                    method = "login";
                    break;
                case StratumCommand.GetJob:
                    method = "getjobtemplate";
                    break;
                case StratumCommand.Keepalive:
                    method = "keepalive";
                    break;
            }
        }

        public void SetParams<T>(T param)
        {
            @params = param;
            //@params = JsonConvert.SerializeObject(param, Formatting.None, new JsonSerializerSettings() { NullValueHandling = NullValueHandling.Ignore });
        }
    }

    public enum StratumCommand
    {
        Status,
        Solution,
        Login,
        GetJob,
        Keepalive
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
   
}
