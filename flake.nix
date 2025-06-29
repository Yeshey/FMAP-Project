# https://github.com/nix-community/dream2nix/tree/main

{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    systems.url = "github:nix-systems/default";
  };

  outputs = { systems, nixpkgs, ... } @ inputs:
  let
    eachSystem = f:
      nixpkgs.lib.genAttrs (import systems) (
        system:
          f (import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          })
      );
  in {
    devShells = eachSystem (pkgs: 
    let
      #lt = pkgs.python311Packages.buildPythonPackage {
      #  pname = "lt_core_news_sm";
      #  version = "3.7.0";
      #  src = pkgs.fetchurl {
      #    url = "https://github.com/explosion/spacy-models/releases/download/pt_core_news_sm-3.7.0/pt_core_news_sm-3.7.0.tar.gz";
      #    hash = "sha256-uG0vywIrvh3/qGkoUskJRZHpPQKZtlxb35L3lArixHw=";
      #  };
      #  buildInputs = with pkgs.python311Packages; [ pipBuildHook ];
      #  dependencies = with pkgs.python311Packages; [ spacy ];
      #};
      #dontCheckPython = drv: drv.overridePythonAttrs (old: { doCheck = false; });
    in {
      default = pkgs.mkShell {
        packages = [
          #pkgs.jupyter-all
          # (pkgs.python312.withPackages (python-pkgs: with python-pkgs; [
          #   # tensorflow
          #   # pytorch
          #   # keras
          #   # pandas
          #   numpy
          #   opencv-python-headless
          #   # jupyter
          #   # matplotlib
          #   # seaborn
          #   # lt
          #   #progressbar
          #   #transformers
          #   #datasets

          #   # script andre
          #   #selenium
          #   #webdriver-manager
          #   #beautifulsoup4
          #   #tqdm

          #   jupyter
          #   jupyter-collaboration
          #   jupyterlab-git

          #   #torch

          #   # Add version overrides for conflicting dependencies
          #   #(protobuf.overridePythonAttrs (old: { version = "4.25.3"; }))
          #   #(typing-extensions.overridePythonAttrs (old: { version = "4.9.0"; }))
          #   #(numpy.override { blas = pkgs.openblasCompat; })
          # ]))

          (pkgs.python312.withPackages (python-pkgs: with python-pkgs; [
            # TensorFlow with CUDA support for GPU acceleration
            tensorflowWithCuda
            #tensorflow-tensorboard ???
            # tensorboard
            
            # Deep Learning and RL Libraries
            stable-baselines3
            gymnasium
            ale-py
            
            # Data Processing
            numpy
            opencv-python-headless
            imageio
            imageio-ffmpeg
            
            # Jupyter and Development
            jupyter
            jupyter-collaboration
            jupyterlab-git
            #ipython
            #ipykernel
            
            # Additional packages from your requirements
            collections-extended
            
            # Note: Some packages like gymnasium[atari] extras need to be handled differently
            # You may need to install these via pip in a nix-shell if they're not available
          ]))

          pkgs.virtualenv
          pkgs.mesa
          #pkgs.chromedriver
        ];

        shellHook = ''
          echo "Development shell ready!"
          echo "Run with something like:"
          echo "jupyter lab --ip=0.0.0.0 --port=8891 --no-browser --ContentsManager.allow_hidden=True --collaborative"

          echo "when running if you see this on the browser: http://127.0.0.1:8891/tree?token=6c86ed65bba3efd8ebd779bb4263094bbaaa4d9c28a648f4, that means that if you want to use the jupyternotebook server as a kernel for VSC, you need to use 6c86ed65bba3efd8ebd779bb4263094bbaaa4d9c28a648f4 as the password"

          # jupyter notebook
        '';
      };
    });
  };
}
