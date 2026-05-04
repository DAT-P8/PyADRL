{
  description = "PyADRL - Python Area Defence Reinforcement Learning";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      forAllSystems =
        function:
        nixpkgs.lib.genAttrs
          [
            "x86_64-linux"
            "aarch64-linux"
            "x86_64-darwin"
            "aarch64-darwin"
          ]
          (
            system:
            function {
              pkgs = nixpkgs.legacyPackages.${system};
            }
          );
    in
    {
      devShells = forAllSystems (
        { pkgs }:
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.python3
              pkgs.uv
              pkgs.ruff
            ];

            env = {
              UV_PYTHON_DOWNLOADS = "never";
              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
              ];
            };

            shellHook = ''
              echo "Initializing git submodules..."
              git submodule init
              git submodule update

              echo "Syncing Python dependencies with uv..."
              uv sync
              source .venv/bin/activate

              echo ""
              echo "Primary commands:"
              echo "  > sh build.sh # Generate protobuf files"
              echo "  > python3 train.py"
              echo "  > python3 eval.py --restore {model_name}"
            '';
          };
        }
      );
    };
}
