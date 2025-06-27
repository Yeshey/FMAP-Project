# FMAP-Project
"traduzir" para tensorflow/keras o model DQN que está no notebook 9

# Make virtual environment python

- If in NixOS, you should have flakes, activate the environment (`nix develop` or `direnv allow`) and run `sh venvnix.sh`

1. python -m venv .venv --copies
2. (you will need python 3.12, `/usr/local/bin/python3.12 -m venv .venv --copies`)
3. Source the environment:
   1. **Linux:** source .venv/bin/activate
   2. **Windows CMD** .venv\Scripts\activate
   3. **Windows Powershell:** .venv\Scripts\Activate.ps1
4. pip install --upgrade pip
5. pip install -r requirements.txt