using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using GrinProMiner.Models;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Mozkomor.GrinGoldMiner;

namespace GrinProMiner.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ConnectionsController : ControllerBase
    {

        [HttpGet("")]
        public ActionResult<IEnumerable<StratumConnectionInfo>> GetAll()
        {
            var sc1 = ConnectionManager.GetConnectionById(1);
            var c1 = new StratumConnectionInfo()
            {
                Address = sc1.ip,
                Port = sc1.port.ToString(),
                Login = sc1.login,
                Password = sc1.password,
                Status = sc1.IsConnected == true ? "Connected" : "Disconnectd",
                LastCommunication = sc1.lastComm.ToString("yyyy-MM-ddTHH:mm:ssK"),
                LastJob = sc1.CurrentJob?.timestamp.ToString("yyyy-MM-ddTHH:mm:ssK"),
            };

            var sc2 = ConnectionManager.GetConnectionById(2);
            var c2 = new StratumConnectionInfo()
            {
                Address = sc2.ip,
                Port = sc2.port.ToString(),
                Login = sc2.login,
                Password = sc2.password,
                Status = sc2.IsConnected == true ? "Connected" : "Disconnectd",
                LastCommunication = sc2.lastComm.ToString("yyyy-MM-ddTHH:mm:ssK"),
                LastJob = sc2.CurrentJob?.timestamp.ToString("yyyy-MM-ddTHH:mm:ssK"),
            };

            return new List<StratumConnectionInfo>() { c1, c2 };
        }

        [HttpGet("active")]
        public ActionResult<StratumConnectionInfo> GetActive()
        {
            try
            {
                var curr = ConnectionManager.GetCurrConn();
                if (curr != null)
                {
                    bool isFee = ConnectionManager.IsInFee();
                    var ci = new StratumConnectionInfo()
                    {
                        Address = isFee ? "FEE" : curr.ip,
                        Port = isFee ? "FEE" : curr.port.ToString(),
                        Login = isFee ? "FEE" : curr.login,
                        Password = isFee ? "FEE" : 
curr.password,
                        Status = curr.IsConnected == true ? "Connected" : "Disconnectd",
                        LastCommunication = curr.lastComm.ToString("yyyy-MM-ddTHH:mm:ssK"),
                        LastJob = curr.CurrentJob?.timestamp.ToString("yyyy-MM-ddTHH:mm:ssK"),
                    };

                    return ci;
                }
                else
                {
                    return StatusCode(404, $"No active connection.");
                }

            }
            catch (Exception ex)
            {
                return StatusCode(500, $"ERROR while getting active connection. {ex.Message}");
            }
        }

        [HttpPost("active")]
        public ActionResult SetActive(Mozkomor.GrinGoldMiner.Connection connection)
        {
            var r = ConnectionManager.SetConnection(connection);

            if (r == "ok")
                return Ok("New connection is active. Check by calling /api/connections/active");
            else
                return StatusCode(500, r);
        }
    }
}