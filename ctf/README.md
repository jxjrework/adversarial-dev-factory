# CTF Toolkit

This is modified from the toolkit for the
[Competition on Adversarial Attacks and Defenses 2018](http://caad.geekpwn.org/) CTF @ LV

## Commands
Please make sure your current directory is at architect
Please make sure permissons have been given to all .sh files by
$ sudo chmod 777 \*.sh

### Operator Level
[1] load attack

```bash
./load_attack.sh
```

[2] watchdog attack (automatic monitor & attack)

```bash
./watchdog_attack.sh
```

[3] send all attack (manual full attack)

```bash
./send_attack.sh
```

[4] single-image attack (manual attack with single png)

```bash
./single_attack.sh
```

[5] display on screen collected results of attacks

```bash
./show_result.sh
```

[6] save current target_result.csv file to history folder and clean up the table

```bash
./clean_result.sh
```

#### Debugging Level
start python2.7 by
$ python
from NWS_Attack import *
