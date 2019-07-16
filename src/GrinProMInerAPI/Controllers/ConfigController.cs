using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Mozkomor.GrinGoldMiner;

namespace GrinProMiner.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ConfigController : ControllerBase
    {
        [HttpGet]
        public ActionResult<Config> Get()
        {
            return GrinProMInerAPI.Program.config;
        }

        [HttpPost]
        public ActionResult Post(Config config)
        {
            var path = GrinProMInerAPI.Program.WriteConfigToDisk(config);
            return Ok($"config saved in {path}, will be activated next time miner is started");

            ///TODO Re-Init Workers and connections / how to reinit workers?
            //GrinProMInerAPI.Program.ChangeRemoteTerminate = true; //will change remote dashboard ping - what if it wasnt running in the first run? then there is no loop active
            //Task.Delay(1000).Wait();
            //ConnectionManager.CloseAll();
            //Task.Delay(1000).Wait();
            //Logger.SetLogOptions(config.LogOptions);
            //WorkerManager.Init(config);
            //ConnectionManager.Init(config);
        }
    }
}