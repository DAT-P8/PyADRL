{
  description = "PyADRL - Python Area Defence Reinforcement Learning";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      pyproject-nix,
      ...
    }:
    let
      # Loads pyproject.toml into a high-level project representation
      project = pyproject-nix.lib.project.loadPyproject { projectRoot = ./.; };

      # Supported systems
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
      # Create a development shell containing dependencies from `pyproject.toml`
      devShells = forAllSystems (
        { pkgs }:
        let
          # We are using the default nixpkgs Python3 interpreter & package set.
          python = pkgs.python3;

          # Returns a function that can be passed to `python.withPackages`
          arg = project.renderers.withPackages { inherit python; };

          # Returns a wrapped environment (virtualenv like) with all our packages
          pythonEnv = python.withPackages arg;
        in
        {
          default = pkgs.mkShell {
            packages = [
              pythonEnv
              pkgs.ruff
            ];
          };
        }
      );
    };
}
