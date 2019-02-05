// Grin Gold Miner https://github.com/urza/GrinGoldMiner
// Copyright (c) 2018 Lukas Kubicek - urza
// Copyright (c) 2018 Jiri Vadura - photon

using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using SharedSerialization;
using System;
using System.Collections.Concurrent;
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
    public class StratumConnet
    {
        public string ip;
        public int port;
        public string login;
        public string password;
        public byte id;

        public DateTime lastShare = DateTime.Now;
        public volatile uint totalShares = 0;
        public volatile uint sharesTooLate = 0;
        public volatile uint sharesRejected = 0;
        public volatile uint sharesAccepted = 0;

        /// <summary>
        /// is listening to TCP client in listener loop
        /// </summary>
        public volatile bool IsConnected;

        /// <summary>
        /// set to true to prevent connecting with invalid login (may "block" secondary connection by connecting and then playing dead)
        /// </summary>
        public bool hasInvalidLogin = false;

        private TcpClient client;
        private NetworkStream stream;
        private SslStream streamTLS;
        private StreamReader reader;
        private Task watchdog = null;
        private Task listener = null;
        private Task sender = null;
        public AutoResetEvent flushToStratum = new AutoResetEvent(false);
        public ConcurrentQueue<StratumRpcRequest> solutionQueue = new ConcurrentQueue<StratumRpcRequest>();
        public Job CurrentJob = null;
        public Job PrevJob = null;

        public DateTime lastComm = DateTime.Now;
        public int BadPacketCnt = 0;
        private volatile bool terminated = false;
        private int mined = 0;
        //internal Stats statistics;
        

        public Action ReconnectAction { get; internal set; }
        private bool UseSsl;

        /// <summary>
        /// Create instance, but not connect yet.
        /// </summary>
        public StratumConnet(string _ip, int _port, byte _id, string _login, string _pwd, bool _ssl = false)
        {
            this.ip = _ip;
            this.port = _port;
            id = _id;
            login = _login;
            password = _pwd;
            UseSsl = _ssl;

            IsConnected = false;
        }

        /// <summary>
        /// connect and ask for job
        /// </summary>
        public void Connect()
        {
            if (hasInvalidLogin)
            {
                IsConnected = false;
                Logger.Log(LogLevel.WARNING, $"Will NOT connect to {ip}:{port} as {login}, server reported invalid login. Please check your connection login details.");
                return;
            }

            try
            {
                if (client != null)
                    client.Dispose();

                System.Net.ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12 | SecurityProtocolType.Tls11 | SecurityProtocolType.Tls;

                Logger.Log(LogLevel.DEBUG, "Connecting to : " + ip);

                client = new TcpClient(ip, port);

                if (client.Connected)
                {
                    Logger.Log(LogLevel.INFO, $"Connected to node {ip}.");

                    BadPacketCnt = 0;
                    IsConnected = true;
                    terminated = false;
                    if (!UseSsl)
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

                    lastComm = DateTime.Now;

                    if (watchdog == null || watchdog.Status == TaskStatus.RanToCompletion || watchdog.Status == TaskStatus.Faulted || watchdog.Status == TaskStatus.Canceled)
                        watchdog = Task.Factory.StartNew(() => { DisconnectMonitor(); }, TaskCreationOptions.LongRunning);

                    if (listener == null || listener.Status == TaskStatus.RanToCompletion || listener.Status == TaskStatus.Faulted || listener.Status == TaskStatus.Canceled)
                        listener = Task.Factory.StartNew(() => { Listen(); }, TaskCreationOptions.LongRunning);
                    
                    if (sender == null || sender.Status == TaskStatus.RanToCompletion || sender.Status == TaskStatus.Faulted || sender.Status == TaskStatus.Canceled)
                        sender = Task.Factory.StartNew(() => { Sender(); }, TaskCreationOptions.LongRunning);
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
            if ((sslPolicyErrors == SslPolicyErrors.None) /*|| (sslPolicyErrors == SslPolicyErrors.RemoteCertificateNameMismatch)*/)
                return true;

            Logger.Log(LogLevel.ERROR, $"Certificate error: {sslPolicyErrors}");

            // Do not allow this client to communicate with unauthenticated servers.
            return false;
        }

        /// <summary>
        /// from what stratum connection originated job, will be send to workers and back with solution
        /// </summary>
        /// <returns></returns>
        private Episode GetJobOrigine()
        {
            if (id == 1 || id == 2) return Episode.user;
            if (id == 3 || id == 4) return Episode.mf;
            if (id == 5 || id == 6) return Episode.gf;
            return Episode.user;
        }

        /// <summary>
        /// waiting for messages from stream and reading them
        /// </summary>
        public void Listen()
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
                        Logger.Log(LogLevel.DEBUG, $"(sc id {id}): Epmty read from reader");
                        BadPacketCnt++;
                        continue;
                    }

                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine();
                    Logger.Log(LogLevel.DEBUG, $"(sc id {id}):TCP IN: {message} {Environment.NewLine}");
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
                                    CurrentJob.origin = GetJobOrigine();
                                    if (CurrentJob != null && CurrentJob.pre_pow != null && CurrentJob.pre_pow != "")
                                    {
                                        lastComm = DateTime.Now;
                                        if (ConnectionManager.IsConnectionCurrent(id))
                                            PushJobToWorkers();
                                    }
                                }
                                break;
                            case "job":
                                PrevJob = CurrentJob ?? null;
                                CurrentJob = new Job(JsonConvert.DeserializeObject<JobTemplate>(para, new JsonSerializerSettings() { NullValueHandling = NullValueHandling.Ignore }));
                                CurrentJob.origin = GetJobOrigine();
                                if (CurrentJob != null && CurrentJob.pre_pow != null && CurrentJob.pre_pow != "")
                                {
                                    if (CurrentJob.scale != 1)
                                    {
                                        lastComm = DateTime.Now;
                                        if (ConnectionManager.IsConnectionCurrent(id))
                                            PushJobToWorkers();
                                    }
                                    else
                                        Logger.Log(LogLevel.WARNING, $"Incorrect pre_pow: {CurrentJob.pre_pow}");
                                }
                                break;
                            case "submit":
                                if (msg.ContainsKey("result") && msg["result"].ToString() == "ok")
                                {
                                    Console.ForegroundColor = ConsoleColor.Cyan;
                                    Logger.Log(LogLevel.INFO, $"(sc id {id}):Share accepted");
                                    sharesAccepted++;
                                    Console.ResetColor();
                                }
                                else if (msg.ContainsKey("result") && msg["result"].ToString().StartsWith("blockfound"))
                                {
                                    Console.ForegroundColor = ConsoleColor.Cyan;
                                    //Console.WriteLine("###################################");
                                    Logger.Log(LogLevel.INFO, "######  Block mined!  #" + (++mined).ToString("D4") + "  ######"); // 8 chars
                                    //Console.WriteLine("###################################");
                                    Console.ResetColor();
                                    //statistics.mined++;
                                    sharesAccepted++;
                                }
                                if(msg.ContainsKey("error"))
                                {
                                    //phooton -32501, "Share rejected due to low difficulty" and -32504 "Shares rejected: duplicates" would that be okay for you
                                    try
                                    {
                                        var code = (int)msg["error"]["code"];
                                        switch (code)
                                        {
                                            case -32503:
                                                Logger.Log(LogLevel.WARNING, "Solution submitted too late");
                                                sharesTooLate++;
                                                break;
                                            case -32502:
                                                Logger.Log(LogLevel.WARNING, "Failed to validate solution");
                                                sharesRejected++;
                                                break;
                                            case -32501:
                                                Logger.Log(LogLevel.WARNING, "Share rejected due to low difficulty");
                                                sharesRejected++;
                                                break;
                                            default:
                                                Logger.Log(LogLevel.WARNING, "Stratum " + (string)msg["error"]["message"]);
                                                break;

                                        }
                                    }
                                    catch { }
                                }
                                break;
                            case "login":
                                //{"id":"Stratum","jsonrpc":"2.0","method":"login","error":{"code":-32501,"message":"invalid login format"}} 
                                    if (msg.ContainsKey("error"))
                                    {
                                        try
                                        {
                                            var errormsg = (string)msg["error"]["message"];
                                            if (!string.IsNullOrEmpty(errormsg))
                                                Logger.Log(LogLevel.ERROR, $"Stratum {ip}:{port} " + errormsg);
                                        }
                                        catch { }
                                        try
                                        {
                                            var code = (int)msg["error"]["code"];
                                            if (code == -32501)
                                            {
                                                terminated = true;
                                                IsConnected = false;
                                                hasInvalidLogin = true;
                                                StratumClose();
                                                Logger.Log(LogLevel.ERROR, "STRATUM INVALID LOGIN ERROR, CLOSING CONNECTION");
                                                Task.Run(() => Task.Delay(2000).ContinueWith(_ => ReconnectAction()));
                                            }
                                        }
                                        catch { }
                                    }
                                //{"id":"Stratum","jsonrpc":"2.0","method":"login","result":"ok","error":null} 
                                if (msg.ContainsKey("result") && msg["result"].ToString() == "ok")
                                {
                                    Logger.Log(LogLevel.INFO, $"Stratum {ip}:{port} login ok");
                                }
                                    break;
                            default:
                                if (method != "keepalive" && !string.IsNullOrEmpty(para))
                                    Logger.Log(LogLevel.INFO, para);
                                if(msg.ContainsKey("error"))
                                {
                                    try
                                    {
                                        var errormsg = (string)msg["error"]["message"];
                                        if (!string.IsNullOrEmpty(errormsg))
                                            Logger.Log(LogLevel.WARNING, "Stratum " + errormsg);
                                    }
                                    catch { }
                                }
                                break;
                        }


                    }
                    catch (System.IO.IOException ex)
                    {
                        Logger.Log(LogLevel.DEBUG, "Catched Socket exeption in listener: " + ex.Message);
                    }
                    catch (System.Net.Sockets.SocketException ex)
                    {
                        Logger.Log(LogLevel.DEBUG, "Catched Socket exeption in listener: " + ex.Message);
                    }
                    catch (Exception ex)
                    {
                        Logger.Log(ex);
                    }
                }
                IsConnected = false; //more like "IsListening" 
                Logger.Log(LogLevel.DEBUG, $"Listener dropped for stratum connection id {id} on thread {Environment.CurrentManagedThreadId}");
                //listenerCancel.ThrowIfCancellationRequested();

            }
            catch (System.Net.Sockets.SocketException ex)
            {
                IsConnected = false;
                Logger.Log(LogLevel.DEBUG, $"(sc id {id}):Catched Socket exeption in listener: " + ex.Message);
            }
            catch (System.IO.IOException ex)
            {
                if (IsConnected == false)
                {
                    Logger.Log(LogLevel.DEBUG, "Catched Socket exeption in listener: " + ex.Message);
                }

                else
                {

                    IsConnected = false;

                    if (ex.InnerException != null && ex.InnerException is SocketException)
                    {
                        var errc1 = ((SocketException)ex.InnerException).ErrorCode;
                        //var errc2 = ((SocketException)ex.InnerException).NativeErrorCode;
                        //var errc3 = ((SocketException)ex.InnerException).SocketErrorCode;
                        // https://docs.microsoft.com/en-us/windows/desktop/winsock/windows-sockets-error-codes-2

                        if (errc1 == 10053) //WSAECONNABORTED - happens when ssl and port combination not supported by pool
                            Logger.Log(LogLevel.ERROR, "Check that your stratum connection details are correct. Ssl (true/false) and port must be supported by pool in this combination.");
                    }
                }
            }
            catch (Exception ex)
            {
                IsConnected = false;
                Logger.Log(ex);
            }
        }

        internal bool PushJobToWorkers()
        {
                Logger.Log(LogLevel.DEBUG, $"({ConnectionManager.solutionCounter}) PushJobToWorkers: sc id {id}, job {CurrentJob.jobID}, job origine {CurrentJob.origin} job timestamp {CurrentJob.timestamp}");
                WorkerManager.newJobReceived(CurrentJob);

                return CurrentJob.pre_pow != "";
        }

        //close connections
        public void StratumClose()
        {
            try
            {
                terminated = true;
                IsConnected = false;
                flushToStratum.Set();

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
                
                listener = null;
                sender = null;
                //watchdog = null;

                Logger.Log(LogLevel.DEBUG, $"Closed connection id {id}");
            }
            catch (System.IO.IOException ex)
            {
                Logger.Log(LogLevel.DEBUG, "Catched Socket exeption in StratumClose: " + ex.Message);
            }
            catch (System.Net.Sockets.SocketException ex)
            {
                Logger.Log(LogLevel.DEBUG, $"(sc id {id}):Catched Socket exeption in StratumClose: " + ex.Message);
            }
            catch (Exception ex)
            {
                Logger.Log(ex);
            }
        }

        /// <summary>
        /// watchdog, if we are disconnected, try to reconnect
        /// </summary>
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
                        StratumClose();
                        ReconnectAction();
                    }
                    else
                    {
                        //we are connected, but current job is too old, try to reconnect(so both primary and secondary connections get tried)
                        if ((DateTime.Now - lastComm) > TimeSpan.FromMinutes(10))
                        {
                            Logger.Log(LogLevel.DEBUG, $", SC ID {id} last comm is too old ({lastComm.ToString()}), Reconnecting from DisconnectMonitor.");
                            StratumClose();
                            ReconnectAction(); //call parent => try both connections
                        }
                    }
                }
                catch (System.IO.IOException ex)
                {
                    Logger.Log(LogLevel.DEBUG, "Catched Socket exeption in DisconnectMonitor: " + ex.Message);
                }
                catch (System.Net.Sockets.SocketException ex)
                {
                    Logger.Log(LogLevel.DEBUG, $"(sc id {id}):Catched Socket exeption in DisconnectMonitor: " + ex.Message);
                }
                catch (Exception ex) { Logger.Log(ex); }

                Task.Delay(5000).Wait();
            }
        }

        /// <summary>
        /// Send serialized class into tcp connection
        /// login, getjob...
        /// </summary>
        public bool GrinSend<T>(T message)
        {
            try
            {
                string output = JsonConvert.SerializeObject(message, Formatting.None, new JsonSerializerSettings() { NullValueHandling = NullValueHandling.Ignore });

                Console.ForegroundColor = ConsoleColor.DarkGreen;
                Logger.Log(LogLevel.DEBUG, $"(sc id {id}): TCP OUT: {output} {Environment.NewLine}");
                Console.ResetColor();

                byte[] bmsg = Encoding.UTF8.GetBytes(output + "\n");

                if (streamTLS != null)
                {
                    lock (streamTLS)
                    {
                        if (streamTLS.CanWrite)
                        {
                            streamTLS.Write(bmsg, 0, bmsg.Length);
                            streamTLS.FlushAsync();
                        }
                        else
                        {
                            IsConnected = false;
                            Logger.Log(LogLevel.DEBUG,$" !! streamTLS.CanWrite == false, disconnecting");
                        }
                    }
                }
                else
                {
                    lock (stream)
                    {
                        if (stream.CanWrite)
                        {
                            stream.Write(bmsg, 0, bmsg.Length);
                            stream.FlushAsync();
                        }
                        else
                        {
                            IsConnected = false;
                            Logger.Log(LogLevel.DEBUG, $" !! stream.CanWrite == false, disconnecting");
                        }
                    }
                }

                return true;
            }
            catch (Exception ex)
            {
                IsConnected = false;
                Logger.Log(ex);
                return false;
            }
        }

        internal void SendSolution(Solution activeSolution)
        {
            try
            {
                // difficulty check here
                if (activeSolution.CheckDifficulty())
                {
                    SubmitParams pow = new SubmitParams() { height = activeSolution.job.height, nonce = activeSolution.job.nonce, pow = activeSolution.nonces.ToList(), job_id = activeSolution.job.jobID };
                    StratumRpcRequest request = new StratumRpcRequest(StratumCommand.Solution);
                    request.SetParams(pow);

                    //if (GrinSend<StratumRpcRequest>(request))
                    lock (solutionQueue)
                    {
                        ///use concurent queue
                        solutionQueue.Enqueue(request);
                        flushToStratum.Set();
                        Logger.Log(LogLevel.DEBUG, $"SOL-{activeSolution.job.hnonce} OUT {DateTime.Now.ToString("mm:ss.FFF")}");
                    }
                    totalShares++;
                    lastShare = DateTime.Now;
                }
                else
                {
                    // low difficulty share
                }
            }
            catch(Exception ex) { Logger.Log(ex); }
        }

        /// <summary>
        /// To prevent writing to stream at the same time from two or more workers
        /// when two solutions are found closely after each other (more cards, chance..)
        /// we instead just put solutions to concurrent queue
        /// and then take them one by one out and send to stream
        /// </summary>
        private void Sender()
        {
            while (!terminated)
            {
                flushToStratum.WaitOne(); //this is just non-blocking waiting for someone somewhere to call flushToStratum.Set()

                //take the message first out from queue and send it
                while (!terminated && solutionQueue.TryDequeue(out StratumRpcRequest r))
                {
                    GrinSend<StratumRpcRequest>(r);
                }
            }
        }

        internal void KeepAlive()
        {
            try
            {
                StratumRpcRequest request = new StratumRpcRequest(StratumCommand.Keepalive);

                if (GrinSend<StratumRpcRequest>(request))
                {
                    //Logger.Log(LogLevel.DEBUG, $"keepalive sent for connection id {id}");
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
                    //Logger.Log(LogLevel.DEBUG, $"Login sent for connection id {id}.");
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
                    //Logger.Log(LogLevel.DEBUG, $"job request sent for connection id {id}");
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
