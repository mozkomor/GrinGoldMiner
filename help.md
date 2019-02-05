
# GGM Features in CLI and Config


### Switch between static TUI (text user interface) and rolling console

 by pressing key "L" (lowercase and uppercase both should work)

### TUI not refreshing?
Try pressing enter in the console window, on Windows the console can be sometimes "paused" by clicking in it

 ### Load config from different location:

    Linux: ./GrinGoldMinerCLI configpath=/absolute/path/to/directory
    Windows: GrinGoldMinerCLI.exe configpath=C:\absolute\path\to\directory
path must be absolute and must exists, config will be created or loaded in this location

 ### Start withou TUI (rolling console only)

    Linux: ./GrinGoldMinerCLI mode=rolling
    Windows: GrinGoldMinerCLI.exe mode=rolling
    

### Define backup pool in case the primary pool is down:
In your config.xml fill the SecondaryConnection details to another pool with your real account, for example:
```
<PrimaryConnection>
    <ConnectionAddress>eu-west-stratum.grinmint.com</ConnectionAddress>
    <ConnectionPort>4416</ConnectionPort>
    <Ssl>true</Ssl>
    <Login>your@email.com/rig1</Login>
    <Password>password</Password>
  </PrimaryConnection>
  <SecondaryConnection>
    <ConnectionAddress>eu.stratum.grin-pool.org</ConnectionAddress>
    <ConnectionPort>3416</ConnectionPort>
    <Ssl>false</Ssl>
    <Login>yourloginhere</Login>
    <Password></Password>
  </SecondaryConnection>
```
### Password in config
Some pools require password every time, some pools only when you join for the first time and then can be blank, other pools  require that you first create account on their web, yet other pools don't require password at all and treat this field differently. Always be sure you understand the pool's policy with "password" field. If you are deleting content of password tag from config.xml (because your pool does not require it) be sure you have it stored somewhere else safely.

### Define what you see in console or what gets logged into files
Can be set in config.xml in 
```
<FileMinimumLogLevel>INFO</FileMinimumLogLevel>
<ConsoleMinimumLogLevel>INFO</ConsoleMinimumLogLevel>
```
Accepted values are DEBUG, INFO, WARNING, ERROR. 
DEBUG will slow down your mining and fill your disk really quickly.

### Offload some work from CPU to GPU

    <CPUOffloadValue>0</CPUOffloadValue>
 
 value is integer between 5 and 90 
 Experimental for Rigs with more cards and really slow CPUs.   
   
### Sample config.xml with comments
[commented config.xml](/src/Master/config.xml)





