{
  description = "PyADRL - Python Area Defence Reinforcement Learning";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    nixpkgs,
    pyproject-nix,
    ...
  }: let
    project = pyproject-nix.lib.project.loadPyproject {projectRoot = ./.;};

    forAllSystems = function:
      nixpkgs.lib.genAttrs
      ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"]
      (
        system:
          function {
            # Import nixpkgs with allowUnfree — required for CUDA
            pkgs = import nixpkgs {
              inherit system;
              config.allowUnfree = true;
            };
          }
      );
  in {
    devShells = forAllSystems (
      {pkgs}: let
        python = pkgs.python3.override {
          packageOverrides = final: prev: {
            torch =
              if pkgs.stdenv.isLinux
              then prev.torch.override {triton = prev.triton-cuda;}
              else prev.torch;
          };
        };

        arg = project.renderers.withPackages {inherit python;};
        pythonEnv = python.withPackages arg;
      in {
        default = pkgs.mkShell {
          packages = [pythonEnv pkgs.ruff];
          shellHook = ''
            echo "Initializing git submodules..."
            git submodule init
            git submodule update

            echo ""
            echo "Primary commands:"
            echo "  > sh build.sh # Generate protobuf files"
            echo "  > python3 main.py --train {name}"
            echo "  > python3 main.py --test {name} --checkpoints ./checkpoints/iter_{num}"
          '';
        };
      }
    );
  };
}
