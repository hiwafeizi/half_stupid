0. Overview

This project uses:

* **Minecraft Java + Forge + Malmo mod** for the world
* **Malmo Python bindings** (native `.pyd` file) for agents
* A **local conda environment inside the repo** for isolation

The exact Malmo build we are using is:

`Malmo-0.37.0-Windows-64bit_withBoost_Python3.7`

This distribution includes:

* `Python_Examples/` with many example scripts
* `MalmoPython.pyd` and `MalmoPython.lib` (the native Python extension)
* No `setup.py`, no `MalmoPython/` package folder
* So you do not install it via `pip install -e .`, you import the `.pyd` directly

We also have the Malmo mod jar (`MalmoMod-0.37.0.jar`) in the Minecraft mods folder to make Malmo show up inside the game.

Anything about versions, such as “Malmo 0.37.0 is built for Minecraft 1.11.2,” should be fact-checked once against the official Malmo docs.

---

## 1. Prerequisites

You need:

1. **Windows 64 bit**
2. **Java** installed (Java 8 is usually recommended for older Minecraft builds; please fact-check this with Minecraft and Malmo docs)
3. **Minecraft Java Edition** (not Bedrock)
4. **Anaconda or Miniconda** installed
5. The Malmo build you are using:

   `Malmo-0.37.0-Windows-64bit_withBoost_Python3.7` (already downloaded in your case)

You also need a working internet connection for initial installs.

---

## 2. Minecraft + Forge + Malmo mod

### 2.1 Install Minecraft Java

* Install the **Minecraft Java Edition** launcher from the official Minecraft site, log in, and run it at least once.
* In the launcher, create or choose an installation for **Minecraft 1.11.2** and run it once so Minecraft creates the `1.11.2` version folder.
  * The exact version to use with Malmo 0.37.0 should be fact-checked in the Malmo docs.

### 2.2 Install Forge for 1.11.2

1. Download the **Forge 1.11.2 installer** (the `.jar` installer, not the universal jar or exe).
2. Run the installer:
   * Double click the `.jar`
   * Choose `Install client`
   * Click OK
   * It should end with something like “Successfully installed Forge”.
3. Open the Minecraft launcher again:
   * Go to **Installations**
   * You should now see something like `forge 1.11.2` (name may vary)
   * If you do not see it, add a new installation and choose a version name that includes `1.11.2-forge...`

**Common issues and tips:**

* If double clicking the `.jar` does nothing, Java is not correctly installed or associated with `.jar` files. In that case, run it from a terminal using:
  ```powershell
  java -jar forge-1.11.2-...-installer.jar
  ```
* If you do not see `Install client` inside the Forge installer, you probably never ran Minecraft 1.11.2 once before. Run 1.11.2 once and try again.

### 2.3 Generate `mods` folder and add Malmo mod

1. In the launcher, choose the **Forge 1.11.2** profile and click Play.
2. Once Minecraft loads to the main menu, close it again. This run creates the `mods` folder.

The `mods` folder path is typically:

```text
C:\Users\<YOUR_USER>\AppData\Roaming\.minecraft\mods
```

3. Copy `MalmoMod-0.37.0.jar` into that `mods` folder.

### 2.4 Verify Malmo mod is loaded

1. Launch Minecraft with the Forge 1.11.2 profile again.
2. On the main menu, click  **Mods** .
3. You should now see something like:
   `Malmo Mod 0.37.0`

If it is not there:

* Check that you really launched the Forge profile, not “latest release”.
* Check that the mod jar is in `.minecraft/mods` (no subfolder).
* Check that the file extension is `.jar` and not `.jar.zip` or similar.

---

## 3. Project-local conda environment (inside the repo)

We keep all Python dependencies inside the repo so collaborators do not mess with global environments.

Assume your project root is:

```text
C:\github\half_stupid
```

### 3.1 Create the env inside the folder

Open **Anaconda Prompt** or PowerShell and run:

```powershell
cd C:\github\half_stupid
conda create --prefix .\env python=3.7
```

Adjust Python version if needed, but 3.7 is typically used with these older bindings. Please fact-check with your actual Malmo build if needed.

### 3.2 Activate the env

In PowerShell:

```powershell
conda activate C:\github\half_stupid\env
```

You should see the prompt change to:

```text
(env) PS C:\github\half_stupid>
```

**Important tip:**

* Do not type the `$` you sometimes see in documentation examples.

  In PowerShell you just type `conda activate ...`, not `$ conda activate ...`.
* If `conda` is not recognized in PowerShell, run:

  ```powershell
  conda init powershell
  ```

  then close and reopen PowerShell.

---

## 4. Using the Malmo Python bindings (`MalmoPython.pyd`)

In your Malmo `Python_Examples` folder you have:

* `MalmoPython.pyd`
* `MalmoPython.lib`
* `malmoutils.py`
* many example scripts like `tutorial_1.py`, `run_mission.py`, etc.

This means Malmo’s Python API is provided as a **compiled module** called `MalmoPython`, not as a package `malmo` you install via pip.

The example scripts in that folder normally do:

```python
import MalmoPython
```

or:

```python
from MalmoPython import AgentHost
```

So the goal is simply:  **make sure Python can import `MalmoPython`** .

There are two main ways to do that.

### Option A: Use `PYTHONPATH` (easy, recommended for development)

1. Keep the Malmo folder somewhere stable, for example:
   ```text
   C:\Tools\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\
   ```
2. Inside your project env, set `PYTHONPATH` so Python also searches the Malmo `Python_Examples` folder.

In PowerShell (for the current session):

```powershell
conda activate C:\github\half_stupid\env
$env:PYTHONPATH = "C:\Users\hiwa\Downloads\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\Python_Examples"
```

Now test:

```powershell
python -c "import MalmoPython; print('OK')"
```

You should see:

```text
OK
```

If that works, you can import `MalmoPython` from your project code.

To make this permanent for each dev, you can:

* Add this `PYTHONPATH` line to a small `scripts\activate_env.ps1` that people run before development, or
* Ask people to set a user environment variable manually (document the exact path to use).

### Option B: Copy the module into your env’s `site-packages` (more permanent)

If you prefer not to deal with environment variables, you can copy the files into the conda env.

1. Find the site-packages path for your local env:

```powershell
conda activate C:\github\half_stupid\env
python -c "import site; print(site.getsitepackages())"
```

You should see something like:

```text
['C:\\github\\half_stupid\\env\\Lib\\site-packages', ...]
```

2. Copy these files from your Malmo `Python_Examples` folder:

* `MalmoPython.pyd`
* `MalmoPython.lib`
* `malmoutils.py`

into:

```text
C:\github\half_stupid\env\Lib\site-packages
```

3. Test again:

```powershell
python -c "import MalmoPython; print('OK')"
```

If that prints `OK`, the bindings are installed for this env.

---

## 5. Launching Minecraft with the Malmo mod

There are two ways you may be doing this:

1. With the normal Minecraft launcher + Forge profile (what you already set up)
2. With a Malmo launch script such as `launchClient.bat` if you also have the full Malmo platform zip (not just the withBoost build)

Given your current setup, you are already doing the Forge profile approach:

* Make sure in the launcher you select the **Forge 1.11.2** installation.
* Click Play.
* Create or open a world.
* Leave Minecraft open in the background with the world loaded.

MalmoPython will then connect to the running Minecraft instance using the port configured by the mod. For the example scripts in `Python_Examples`, this is usually `127.0.0.1` and a default port configured in the mission XML. You may need to check your example script for the exact port.

If you later switch to a Malmo distribution that includes `launchClient.bat`, you can document that as an alternative: run `launchClient.bat` instead of going through the normal launcher.

---

## 6. Running example missions from `Python_Examples`

To verify everything works:

1. Activate the env and set `PYTHONPATH` (or ensure you copied files into `site-packages`):

```powershell
conda activate C:\github\half_stupid\env
$env:PYTHONPATH = "C:\Users\hiwa\Downloads\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\Python_Examples"
```

2. Launch Minecraft with Forge + Malmo mod and load a world.
3. In another terminal, go to the `Python_Examples` folder:

```powershell
cd "C:\Users\hiwa\Downloads\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\Python_Examples"
```

4. Run a tutorial:

```powershell
python tutorial_1.py
```

or:

```powershell
python run_mission.py
```

You should see:

* The script printing something like “Waiting for mission to start”
* Minecraft briefly starting a mission
* The agent doing something (moving, etc)

If that works, the whole Malmo stack is connected.

---

## 7. Running project code from your repo

In your repo root `C:\github\half_stupid`, you can write a minimal test file such as `test_malmo.py`:

```python
import MalmoPython
import time

def main():
    agent = MalmoPython.AgentHost()
    print("MalmoPython loaded and AgentHost created successfully")

if __name__ == "__main__":
    main()
```

To run it:

```powershell
cd C:\github\half_stupid
conda activate .\env
$env:PYTHONPATH = "C:\Users\hiwa\Downloads\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\Python_Examples"
python test_malmo.py
```

You should see the confirmation message.

Later, when you build your full brain system, all your imports will simply use `import MalmoPython` and the higher level code will live entirely under your repo.

---

## 8. Common issues we hit and tips

You can keep this section as a “pitfalls” area so new contributors skip the pain.

1. **Forge profile vs vanilla profile**
   * If you launch “Latest release” instead of the Forge 1.11.2 profile, the Mods menu will be missing and Malmo will not load.
   * Always confirm the bottom left says something with Forge and the correct Minecraft version.
   * This behavior should be fact-checked if versions change.
2. **No `mods` folder**
   * The `mods` folder only appears after running Minecraft once with Forge.
   * If you do not see `.minecraft/mods`, launch Forge once and then look again.
3. **Mod not detected**
   * Make sure `MalmoMod-0.37.0.jar` is directly under `.minecraft/mods`, not in a subfolder.
   * Confirm the file extension is `.jar`.
4. **Forge installer shows no “Install client”**
   * Usually means you did not run that Minecraft version once before. Run 1.11.2, then rerun the Forge installer.
   * Or you downloaded the wrong Forge file. Check carefully that it is the installer jar. Versions need to be fact-checked.
5. **Java `.jar` does not open**
   * Java is not installed, not in PATH, or `.jar` file associations are broken.
   * Workaround: run in terminal:
     ```powershell
     java -jar forge-1.11.2-...-installer.jar
     ```
6. **PowerShell confusion (`$ conda activate` vs `conda activate`)**
   * The `$` shown in docs is a shell prompt symbol, not part of the command.
   * In PowerShell, you should type `conda activate path_or_name`.
7. **Running `test.py` without `python`**
   * Typing `test.py` alone tries to execute it as a PowerShell command.
   * Use either `python test.py` or `.\test.py`.
8. **`ModuleNotFoundError: No module named 'malmo'`**
   * In this build the correct module is `MalmoPython`, not `malmo`.
   * Example scripts import `MalmoPython`, not `from malmo import MalmoPython`.
   * Make sure your code uses `import MalmoPython` and PYTHONPATH or site-packages is set correctly.
9. **Wrong Malmo package (Mac vs Windows, source vs binary)**
   * Malmo has different packages. Some are source code, some are Mac, some are Windows, some are Python only.
   * The one you are using has `MalmoPython.pyd` under `Python_Examples` and is for Windows.
   * For anyone else setting this up, confirm they download the correct Windows build.
